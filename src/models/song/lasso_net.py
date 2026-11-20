import torch
from itertools import chain


class KGLassoNet(torch.nn.Module):
    TOTAL_SMRI_ROIS = 988
    TOTAL_MULTIMODAL_FEATURES = 1976
    ATLAS_ROIS = {
        "AAL1": 116,
        "AAL2": 120,
        "AAL3": 166,
        "Brodmann": 41,
        "Hammersmith": 83,
        "HarvardOxford": 113,
        "Juelich": 103,
        "Brainnetome": 246,
    }

    def build_multimodal_prior_vector(
        self, prior_c: dict, is_multimodal: bool = True
    ) -> torch.Tensor:
        dim = self.TOTAL_MULTIMODAL_FEATURES if is_multimodal else self.TOTAL_SMRI_ROIS
        c = torch.ones(dim, dtype=torch.float32)

        def populate(offset: int):
            curr = offset
            for name, count in self.ATLAS_ROIS.items():
                priors = prior_c.get(name, {})
                for idx in priors.get("c_06", []):
                    if idx < count:
                        c[curr + idx] = 0.6
                for idx in priors.get("c_08", []):
                    if idx < count:
                        c[curr + idx] = 0.8
                curr += count

        populate(offset=0)  # Ramo sMRI
        if is_multimodal:
            populate(offset=self.TOTAL_SMRI_ROIS)  # Ramo PET
        return c

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[list[int]],
        out_dim: int,
        M: int,
        prior_c: torch.Tensor | list,
        multimodal: bool = False,
        dropout: float = 0.0,
        lambda_reg: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.M = M
        self.prior_c = self.build_multimodal_prior_vector(
            prior_c, is_multimodal=multimodal
        )
        self.skip = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.feed_forward = torch.nn.Sequential(
            *chain(
                *(
                    [
                        [
                            torch.nn.Linear(h_0, h_1),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=dropout),
                        ]
                        for (h_0, h_1) in hidden_dims
                    ]
                    + [[torch.nn.Linear(hidden_dims[-1][-1], out_dim)]]
                )
            )
        )
        self.first_dense = self.feed_forward[0]
        self.lambda_reg = lambda_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.feed_forward(x)

    @torch.no_grad()
    def project_hierarchical_constraint(self):
        phi_norms = torch.norm(self.skip.weight, p=2, dim=0)
        max_allowed = self.M * phi_norms
        w1 = self.first_dense.weight.data
        w1_abs = torch.abs(w1)
        for j in range(self.in_dim):
            bound = max_allowed[j].item()
            if bound == 0.0:
                w1[:, j] = 0.0
            else:
                col_max = torch.max(w1_abs[:, j]).item()
                if col_max > bound:
                    w1[:, j] = w1[:, j] * (bound / col_max)

    @torch.no_grad()
    def get_feature_importances(self) -> torch.Tensor:
        phi_norms = torch.norm(self.skip.weight, p=2, dim=0)
        w1_norms = torch.norm(self.first_dense.weight.data, p=2, dim=0)
        return phi_norms + (1.0 / self.M) * w1_norms

    def select_top_features(
        self, k: int = 144
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Seleziona le migliori k feature (144) e crea la Feature-Value Map A_m (12x12)
        per il modulo di diffusione.
        """
        x = int(torch.sqrt(torch.tensor(k)).item())
        assert x**2 == k

        importances = self.get_feature_importances()
        top_scores, top_indices = torch.topk(importances, k=k)
        norm_scores = (top_scores - top_scores.min()) / (
            top_scores.max() - top_scores.min() + 1e-8
        )
        feature_value_map = norm_scores.view(x, x)
        return top_indices, top_scores, feature_value_map

    def get_modality_breakdown(self, top_indices: torch.Tensor) -> dict[str, int]:
        """Conta quante delle 144 feature appartengono a sMRI e quante a PET."""
        idx_list = top_indices.cpu().numpy().tolist()
        smri = sum(1 for i in idx_list if i < self.TOTAL_SMRI_ROIS)
        pet = sum(1 for i in idx_list if i >= self.TOTAL_SMRI_ROIS)
        return {"total": len(idx_list), "smri": smri, "pet": pet}

    def compute_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = torch.nn.functional.cross_entropy(preds, targets)
        phi_norms = torch.norm(
            self.skip.weight, p=2, dim=0
        )  # Magnitudo di ogni feature
        reg_penalty = torch.sum(self.prior_c.to(phi_norms.device) * phi_norms)  # L1 pesato dal prior c
        return ce_loss + self.lambda_reg * reg_penalty, ce_loss, reg_penalty
