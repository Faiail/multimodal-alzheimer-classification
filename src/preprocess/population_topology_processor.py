import os
import torch
import pandas as pd
import numpy as np
from src.utils import ParameterKeys


class PopulationTopologyProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        data_path = self.parameters.get(
            ParameterKeys.DATA,
            self.parameters.get(ParameterKeys.RAW_DATASET),
        )
        self.dataset = pd.read_csv(data_path)
        self.out_dir = self.parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.patient_col = self.parameters.get(ParameterKeys.PATIENT)
        self.use_sex = self.parameters.get(ParameterKeys.SEX, True)
        self.use_apoe4 = self.parameters.get(ParameterKeys.APOE4, True)
        self.use_age = self.parameters.get(ParameterKeys.AGE, False)
        self.age_threshold = self.parameters.get(ParameterKeys.AGE_THRESHOLD, 2.0)
        self.lambda_long = self.parameters.get(ParameterKeys.LAMBDA, 10.0)

    def launch(self):
        self.dataset["node_id"] = np.arange(len(self.dataset), dtype=np.int64)
        n = len(self.dataset)
        adj = np.zeros((n, n), dtype=np.float32)
        if self.use_sex:
            sex = (
                self.dataset["PTGENDER"]
                .fillna("UNK")
                .astype(str)
                .str.upper()
                .to_numpy()
            )
            valid_s = (sex[:, None] != "UNK") & (sex[None, :] != "UNK")
            adj += ((sex[:, None] == sex[None, :]) & valid_s).astype(np.float32)
        if self.use_apoe4:
            apoe = self.dataset["APOE4"].fillna(-1).to_numpy()
            valid_a = (apoe[:, None] >= 0) & (apoe[None, :] >= 0)
            adj += ((apoe[:, None] == apoe[None, :]) & valid_a).astype(np.float32)
        if self.use_age:
            age = self.dataset["AGE"].to_numpy(dtype=np.float32)
            valid_age = (~np.isnan(age[:, None])) & (~np.isnan(age[None, :]))
            adj += (
                (np.abs(age[:, None] - age[None, :]) < self.age_threshold) & valid_age
            ).astype(np.float32)
        same_ptid = np.zeros((n, n), dtype=bool)
        ptids = self.dataset[self.patient_col].astype(str).to_numpy()
        same_ptid = ptids[:, None] == ptids[None, :]
        adj += self.lambda_long * same_ptid.astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        np.fill_diagonal(same_ptid, False)
        src, dst = np.nonzero(adj)
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        edge_weight = torch.tensor(adj[src, dst].copy(), dtype=torch.float32)
        n_edges = int(edge_index.shape[1])
        n_long_edges = int(same_ptid[src, dst].sum())
        n_unique_ptids = (
            int(self.dataset[self.patient_col].nunique())
            if self.patient_col in self.dataset.columns
            else n
        )
        density = (n_edges / max(n * (n - 1), 1)) * 100.0
        avg_degree = n_edges / max(n, 1)
        del adj, same_ptid
        torch.save(
            {
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "num_nodes": int(n),
            },
            f"{self.out_dir}/topology.pt",
        )
        self.dataset.to_csv(f"{self.out_dir}/population_dataset.csv", index=False)
