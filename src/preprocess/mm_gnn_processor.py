from src.utils import ParameterKeys
import src.data.preprocess as data_pkg
from torch.utils.data import Dataset, DataLoader
from src.data.utils import BatchKeys
from tqdm import tqdm
import os
import torch
import pandas as pd
from torch_geometric.utils import dense_to_sparse
from safetensors.torch import save_file


class MMGraphProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        print("Init general...")
        self._init_general()
        print("Done!")
        print("Init dataset...")
        self._init_data()
        print("Done!")

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL)
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.fold_id = general_parameters.get(ParameterKeys.FOLD)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        self.sex = general_parameters.get(ParameterKeys.SEX)
        self.apoe = general_parameters.get(ParameterKeys.APOE4)
        self.threshold = general_parameters.get(ParameterKeys.THRESHOLD, 0.0)

    def _init_dataset(self, dataset_parameters: dict) -> Dataset:
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_config = dataset_parameters.get(ParameterKeys.CFG, dict())
        return data_pkg.__dict__[dataset_name](**dataset_config)

    def _init_data(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        train_dataset_prameters = dataset_parameters.get(ParameterKeys.TRAIN)
        train_dataset = self._init_dataset(train_dataset_prameters)
        loader_parameters = self.parameters.get(ParameterKeys.LOADER, dict())
        self.train_loader = DataLoader(
            dataset=train_dataset, shuffle=True, **loader_parameters
        )
        population_dataset_parameters = dataset_parameters.get(ParameterKeys.POPULATION)
        self.population_dataset = self._init_dataset(population_dataset_parameters)
        self.population_loader = DataLoader(
            dataset=self.population_dataset, shuffle=False, **loader_parameters
        )

    def get_bar(self, loader: DataLoader, desc: str = "", **kwargs):
        return (
            loader
            if not self.pbar
            else tqdm(
                loader,
                total=len(loader),
                desc=desc,
                **kwargs,
            )
        )

    def get_train_statistics(self):
        mri = None
        pet = None
        bar = self.get_bar(
            loader=self.train_loader, desc="Getting healthy patients statistics"
        )
        for batch in bar:
            batch_mri = batch[BatchKeys.MRI]
            batch_pet = batch[BatchKeys.PET]

            mri = batch_mri if mri is None else torch.cat([mri, batch_mri], dim=0)
            pet = batch_pet if pet is None else torch.cat([pet, batch_pet], dim=0)

        mri_mean = mri.mean(dim=0)
        mri_std = mri.std(dim=0)
        mri_std[mri_std == 0] = 1.0
        mri_centered = mri - mri_mean.unsqueeze(0)
        mri_normed = mri_centered / (mri_centered.norm(dim=0, keepdim=True) + 1e-8)
        mri_M = torch.mm(mri_normed.t(), mri_normed).clamp(-1, 1)

        pet_mean = pet.mean(dim=0)
        pet_std = pet.std(dim=0)
        pet_std[pet_std == 0] = 1.0
        pet_centered = pet - pet_mean.unsqueeze(0)
        pet_normed = pet_centered / (pet_centered.norm(dim=0, keepdim=True) + 1e-8)
        pet_M = torch.mm(pet_normed.t(), pet_normed).clamp(-1, 1)

        return (mri_mean, mri_std, mri_M), (pet_mean, pet_std, pet_M)

    def get_node_features(
        self,
        nc_mean: torch.Tensor,
        nc_std: torch.Tensor,
        nc_M: torch.Tensor,
        modality_key: str,
        desc: str,
    ) -> torch.Tensor:
        bar = self.get_bar(self.population_loader, desc=desc)
        N = len(self.population_loader.dataset)
        R = nc_mean.size(0)
        var_nc = nc_std**2
        denom_mat = torch.sqrt((var_nc.unsqueeze(0) + var_nc.unsqueeze(1)) / 2.0)
        denom_mat = denom_mat.clamp(min=1e-8)
        triu_row, triu_col = torch.triu_indices(R, R, offset=1)
        num_features = len(triu_row)  # 13,695 for R=166
        out_features = torch.empty((N, num_features), dtype=torch.float32)
        cursor = 0
        for batch in bar:
            x = batch[modality_key].float()
            bs = x.size(0)
            # Delta: (f_i - mean_nc_i) -> [B, R]
            delta = x - nc_mean.unsqueeze(0)
            # Eq. 1: Effect size E(i, j) -> [B, R, R]
            num_mat = delta.unsqueeze(2) - delta.unsqueeze(1)
            E_mat = num_mat / denom_mat.unsqueeze(0)
            # Eq. 2: R(i, j) = tanh(E)
            R_mat = torch.tanh(E_mat)
            # Eq. 3: Weighting matrix W = 1 - |R|
            W_mat = 1.0 - torch.abs(R_mat)
            # Eq. 4: Brain network matrix B = W * M_nc
            B_mat = W_mat * nc_M.unsqueeze(0)
            # FIX: Store B_mat (not R_mat!)
            out_features[cursor : cursor + bs] = B_mat[:, triu_row, triu_col].cpu()
            cursor += bs
        return out_features

    def _compute_rbf_sim(self, features: torch.Tensor) -> torch.Tensor:
        """Computes dense [N, N] Gaussian RBF similarity matrix using
        correlation distance rho = 1 - Pearson(u, v).
        """
        N = features.size(0)
        x = torch.nan_to_num(features.float(), nan=0.0)
        # 1. Row-center and L2-normalize -> [N, D]
        centered = x - x.mean(dim=1, keepdim=True)
        normed = centered / centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # 2. Pearson correlation matrix -> [N, N]
        pearson = torch.mm(normed, normed.t()).clamp(-1.0, 1.0)
        rho = 1.0 - pearson  # Correlation distance in [0, 2]
        # 3. Kernel bandwidth sigma = mean off-diagonal distance
        off_diag = ~torch.eye(N, dtype=torch.bool, device=x.device)
        sigma = rho[off_diag].mean().clamp(min=1e-8)
        # 4. Eq. (6): Gaussian RBF kernel
        S = torch.exp(-(rho**2) / (2.0 * sigma**2))
        S.fill_diagonal_(0.0)
        return S

    def get_edge(
        self,
        mri_feats: torch.Tensor,
        pet_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sex = torch.tensor(
            self.population_dataset.dataset[self.sex]
            .astype("category")
            .cat.codes.values,
            dtype=torch.long,
        )
        apoe = torch.tensor(
            pd.to_numeric(self.population_dataset.dataset[self.apoe], errors="coerce")
            .fillna(-1)
            .values,
            dtype=torch.float,
        )

        r_sex = (sex.unsqueeze(0) == sex.unsqueeze(1)).float()
        r_apoe = (apoe.unsqueeze(0) == apoe.unsqueeze(1)).float()
        W_pheno = r_sex + r_apoe

        S_s = self._compute_rbf_sim(mri_feats)  # sMRI
        S_f = self._compute_rbf_sim(pet_feats)  # PET
        S_c = self._compute_rbf_sim(
            torch.cat([mri_feats, pet_feats], dim=1)
        )  # Concatenated

        A_s = S_s * W_pheno
        A_f = S_f * W_pheno
        A_fm = S_c * W_pheno
        A_if = (A_s * A_f) * A_fm
        A_if.fill_diagonal_(0.0)
        A_if[A_if < self.threshold] = 0.0
        edge_index, edge_weight = dense_to_sparse(A_if)
        return edge_index, edge_weight

    def launch(self) -> None:
        print("Launch feature extraction!")
        mri_stats, pet_stats = self.get_train_statistics()
        mri_feats = self.get_node_features(
            *mri_stats, modality_key=BatchKeys.MRI, desc="Get MRI features"
        )
        pet_feats = self.get_node_features(
            *pet_stats, modality_key=BatchKeys.PET, desc="Get PET features"
        )

        edge_index, edge_weight = self.get_edge(
            mri_feats=mri_feats,
            pet_feats=pet_feats,
        )
        save_file(
            {
                BatchKeys.MRI: mri_feats,
                BatchKeys.PET: pet_feats,
                BatchKeys.EDGE_INDEX: edge_index,
                BatchKeys.EDGE_WEIGHT: edge_weight,
            },
            f"{self.out_dir}/graph.safetensors",
        )
