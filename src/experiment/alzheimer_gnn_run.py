import os
from copy import deepcopy
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

import src.losses as losses
import src.models as models
from src.data.batch_utils import (
    build_graph_dataloader,
    fma_dataset_kwargs,
    get_node_feature_dim,
    get_node_feature_dims,
    move_graph_batch_to_device,
)
from src.data.dataset import AlzheimerGraphDataset, HybridAlzheimerDataset
from src.data.preprocessing import FoldFeatureNormalizer
from src.explainability.checkpoint import compute_split_fingerprint
from src.metrics import compute_classification_metrics
from src.utils import ParameterKeys, resolve_data_path, save_json, set_seed


def prepare_alzheimer_dataframe(data_cfg: dict) -> pd.DataFrame:
    """Load and optionally merge imaging ROI features with clinical/diagnostic CSV."""
    features_file = resolve_data_path(
        data_cfg.get("features_file") or data_cfg.get("csv_path")
    )
    clinical_file = resolve_data_path(data_cfg.get("clinical_file"))
    target_col = data_cfg.get("target_col", "target")

    if not features_file or not os.path.exists(features_file):
        raise FileNotFoundError(f"Features CSV not found: '{features_file}'")

    df_img = pd.read_csv(features_file, low_memory=False)

    if clinical_file and os.path.exists(clinical_file) and target_col not in df_img.columns:
        df_clin = pd.read_csv(clinical_file, low_memory=False)
        if "PTID" not in df_img.columns and "subject_id" in df_img.columns:
            df_img["PTID"] = df_img["subject_id"].apply(lambda x: "_".join(str(x).split("_")[:3]))
        merge_cols = [c for c in df_clin.columns if c not in df_img.columns or c == "PTID"]
        df_full = pd.merge(df_img, df_clin[merge_cols], on="PTID", how="inner")
        if "label" in df_full.columns and target_col not in df_full.columns:
            df_full = df_full.rename(columns={"label": target_col})
    else:
        df_full = df_img.copy()
        if "label" in df_full.columns and target_col not in df_full.columns:
            df_full = df_full.rename(columns={"label": target_col})

    df_full = df_full.dropna(subset=[target_col]).copy().reset_index(drop=True)
    df_full[target_col] = df_full[target_col].astype(int)

    for col in ("Gender_Male", "APOE4", "MMSE", "AGE", "PTEDUCAT"):
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna(0)

    return df_full


class AlzheimerGNNRun:
    """StratifiedGroupKFold cross-validation runner for MCI conversion, CN vs AD, and Multiclass CN-MCI-AD."""

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.general = parameters.get(ParameterKeys.GENERAL, {})
        self.out_dir = self.general.get(ParameterKeys.OUT_DIR, "models/alzheimer_gnn")
        os.makedirs(self.out_dir, exist_ok=True)

        device_str = self.general.get(
            ParameterKeys.DEVICE, "cuda" if torch.cuda.is_available() else "cpu"
        )
        if str(device_str).startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)

        self.seed = int(self.general.get(ParameterKeys.SEED, 42))
        set_seed(self.seed)

        self.num_epochs = int(
            self.general.get(
                ParameterKeys.NUM_EPOCHS,
                parameters.get("training", {}).get("epochs", 100),
            )
        )
        self.n_splits = int(
            self.general.get(
                ParameterKeys.N_SPLITS,
                parameters.get("training", {}).get("n_splits", 5),
            )
        )
        self.task = str(self.general.get("task", "multiclass"))
        self.save_plots = bool(self.general.get("save_plots", True))

        loader_cfg = parameters.get(ParameterKeys.LOADER, parameters.get("training", {}))
        self.batch_size = int(loader_cfg.get("batch_size", 16))
        self.num_workers = int(loader_cfg.get("num_workers", 0))

        data_cfg = dict(parameters.get(ParameterKeys.DATA, {}))
        model_section = dict(parameters.get(ParameterKeys.MODEL, {}))
        model_cfg = dict(model_section.get(ParameterKeys.CFG, model_section))

        self.model_name = model_section.get(
            ParameterKeys.NAME, model_cfg.get("type", "GAT_KAN_Classifier")
        )
        self.graph_mode = str(
            data_cfg.get("graph_mode", model_cfg.get("graph_type", "static"))
        ).lower()
        self.num_classes = int(
            model_cfg.get("n_classes", model_cfg.get("out_classes", 2))
        )
        self.tabular_cols: List[str] = list(data_cfg.get("tabular_cols", []))

        self.dataset_kwargs = {
            "csv_path": resolve_data_path(
                data_cfg.get("features_file") or data_cfg.get("csv_path")
            ),
            "coords_path": resolve_data_path(
                data_cfg.get("coords_file", "configs/aal3_coordinates.csv")
            ),
            "target_col": data_cfg.get("target_col", "target"),
            "sigma": float(
                data_cfg.get("sigma", model_cfg.get("spatial_sigma", 25.0))
            ),
            "graph_type": self.graph_mode,
            "knn_k": int(data_cfg.get("knn_k", model_cfg.get("knn_k", 10))),
            "zhang_threshold": float(
                data_cfg.get("zhang_threshold", model_cfg.get("zhang_threshold", 0.5))
            ),
            "graph_mix_alpha": float(
                data_cfg.get("graph_mix_alpha", model_cfg.get("graph_mix_alpha", 0.5))
            ),
            "node_semantics_mode": data_cfg.get(
                "node_semantics_mode", model_cfg.get("node_semantics_mode")
            ),
            "use_semantic_features": bool(
                data_cfg.get(
                    "use_semantic_features",
                    model_cfg.get("use_semantic_features", True),
                )
            ),
            "ontology_lookup_path": resolve_data_path(
                data_cfg.get(
                    "ontology_lookup_file",
                    model_cfg.get("ontology_lookup_path", "configs/aal3_ontology_lookup.csv"),
                )
            ),
            **fma_dataset_kwargs({"model": {**data_cfg, **model_cfg}}),
        }
        if self.dataset_kwargs.get("fma_graph_path"):
            self.dataset_kwargs["fma_graph_path"] = resolve_data_path(
                self.dataset_kwargs["fma_graph_path"]
            )

        self.model_kwargs = {
            "n_clinical_features": len(self.tabular_cols),
            "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
            "n_classes": self.num_classes,
            "dropout": float(model_cfg.get("dropout", 0.3)),
            "use_kan_backbone": bool(model_cfg.get("use_kan_backbone", False)),
            "use_kan_head": bool(model_cfg.get("use_kan_head", True)),
            "n_layers": int(
                model_cfg.get("n_layers", model_cfg.get("num_gnn_layers", 2))
            ),
            "graph_type": self.graph_mode,
        }

        opt_cfg = parameters.get(ParameterKeys.OPTIMIZER, {}).get(
            ParameterKeys.CFG, parameters.get("training", {})
        )
        self.lr = float(opt_cfg.get("lr", opt_cfg.get("learning_rate", 0.0005)))
        self.weight_decay = float(opt_cfg.get("weight_decay", 1e-4))

        es_cfg = parameters.get(ParameterKeys.EARLY_STOP, {}).get(
            ParameterKeys.CFG, parameters.get("training", {})
        )
        self.patience = int(es_cfg.get("patience", 25))

        crit_cfg = parameters.get(ParameterKeys.CRITERION, {})
        self.criterion_name = crit_cfg.get(ParameterKeys.NAME, "FocalLoss")
        self.focal_gamma = float(
            crit_cfg.get(ParameterKeys.CFG, {}).get(
                "gamma", parameters.get("training", {}).get("focal_gamma", 2.0)
            )
        )

    def _build_criterion(self, y_train: np.ndarray) -> torch.nn.Module:
        counts = np.bincount(y_train, minlength=self.num_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        inv_freq = len(y_train) / (self.num_classes * counts)
        alpha = torch.tensor(inv_freq / inv_freq.sum(), dtype=torch.float32, device=self.device)
        if self.criterion_name == "CrossEntropyLoss":
            return losses.CrossEntropyLoss(weight=alpha)
        return losses.FocalLoss(alpha=alpha, gamma=self.focal_gamma)

    def _instantiate_model(self, train_ds) -> torch.nn.Module:
        model_cls = models.__dict__[self.model_name]
        kwargs = deepcopy(self.model_kwargs)
        kwargs["in_features"] = get_node_feature_dim(train_ds)
        kwargs["hetero_metadata"] = getattr(train_ds, "hetero_metadata", None)
        kwargs["hetero_input_dims"] = get_node_feature_dims(train_ds)
        return model_cls(**kwargs).to(self.device), kwargs

    def _forward_batch(self, model: torch.nn.Module, batch_data):
        has_clin = bool(self.tabular_cols)
        if has_clin:
            x_graph, adj, x_clin, y = move_graph_batch_to_device(
                batch_data, self.device, graph_type=self.graph_mode, has_clinical=True
            )
        else:
            x_graph, adj, y = move_graph_batch_to_device(
                batch_data, self.device, graph_type=self.graph_mode, has_clinical=False
            )
            x_clin = None
        logits = model(x_graph, adj, x_clin)
        return logits, y

    def _evaluate_loader(self, model: torch.nn.Module, loader, criterion: torch.nn.Module):
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in loader:
                logits, y = self._forward_batch(model, batch_data)
                loss = criterion(logits, y)
                total_loss += float(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_prob.append(probs)
                y_pred.extend(preds.tolist())
                y_true.extend(y.cpu().numpy().tolist())

        y_prob_arr = np.concatenate(y_prob, axis=0) if y_prob else np.zeros((0, self.num_classes))
        metrics = compute_classification_metrics(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            y_prob=y_prob_arr,
            num_classes=self.num_classes,
        )
        metrics["loss"] = total_loss / max(1, len(loader))
        return metrics, np.asarray(y_true), np.asarray(y_pred), y_prob_arr

    def launch(self) -> Dict[str, float]:
        data_cfg = self.parameters.get(ParameterKeys.DATA, {})
        df_full = prepare_alzheimer_dataframe(data_cfg)
        target_col = self.dataset_kwargs["target_col"]

        if "subject_id" not in df_full.columns:
            if "PTID" in df_full.columns:
                df_full["subject_id"] = df_full["PTID"]
            else:
                df_full["subject_id"] = [f"subj_{i}" for i in range(len(df_full))]

        temp_csv = os.path.join(self.out_dir, "prepared_dataset.csv")
        df_full.to_csv(temp_csv, index=False)

        y_all = df_full[target_col].values
        groups = df_full["subject_id"].values
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        fold_rows = []
        all_y_true, all_y_pred, all_y_prob = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df_full, y_all, groups=groups)):
            fold_num = fold_idx + 1
            print(f"\n--- Fold {fold_num}/{self.n_splits} ({self.model_name} | {self.graph_mode}) ---")

            df_train_raw = df_full.iloc[train_idx].reset_index(drop=True)
            df_val_raw = df_full.iloc[val_idx].reset_index(drop=True)

            normalizer = FoldFeatureNormalizer().fit(df_train_raw)
            df_train = normalizer.transform(df_train_raw)
            df_val = normalizer.transform(df_val_raw)

            reference_stats = None
            if self.graph_mode in ("zhang", "hybrid", "multigraph"):
                cn_df = df_train[df_train[target_col] == 0]
                if len(cn_df) < 10:
                    cn_df = df_train
                ref_kwargs = deepcopy(self.dataset_kwargs)
                ref_kwargs.pop("csv_path", None)
                ref_kwargs["graph_type"] = "static"
                ref_kwargs["precompute_adjacency"] = False
                ref_kwargs["node_semantics_mode"] = "none"
                ref_ds = AlzheimerGraphDataset(df=cn_df, **ref_kwargs)
                reference_stats = ref_ds.get_reference_stats()

            ds_kwargs = deepcopy(self.dataset_kwargs)
            ds_kwargs.pop("csv_path", None)

            if self.tabular_cols:
                train_ds = HybridAlzheimerDataset(
                    tabular_cols=self.tabular_cols,
                    df=df_train,
                    precomputed_stats=reference_stats,
                    **ds_kwargs,
                )
                val_ds = HybridAlzheimerDataset(
                    tabular_cols=self.tabular_cols,
                    df=df_val,
                    precomputed_stats=reference_stats,
                    **ds_kwargs,
                )
            else:
                train_ds = AlzheimerGraphDataset(
                    df=df_train,
                    precomputed_stats=reference_stats,
                    **ds_kwargs,
                )
                val_ds = AlzheimerGraphDataset(
                    df=df_val,
                    precomputed_stats=reference_stats,
                    **ds_kwargs,
                )

            train_loader = build_graph_dataloader(
                train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )
            val_loader = build_graph_dataloader(
                val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )

            model, resolved_model_kwargs = self._instantiate_model(train_ds)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
            criterion = self._build_criterion(df_train[target_col].values)

            best_val_metric = -1.0
            best_state = None
            best_metrics = None
            best_outputs = None
            patience_cnt = 0

            for epoch in range(1, self.num_epochs + 1):
                model.train()
                for batch_data in train_loader:
                    optimizer.zero_grad()
                    logits, y = self._forward_batch(model, batch_data)
                    loss = criterion(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()

                val_metrics, y_t, y_p, y_pr = self._evaluate_loader(model, val_loader, criterion)
                score = val_metrics["balanced_accuracy"]
                if score > best_val_metric:
                    best_val_metric = score
                    best_state = deepcopy(model.state_dict())
                    best_metrics = val_metrics
                    best_outputs = (y_t, y_p, y_pr)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        break

            ckpt_dataset_kwargs = deepcopy(self.dataset_kwargs)
            ckpt_dataset_kwargs["csv_path"] = temp_csv
            ckpt_bundle = {
                "task": self.task,
                "model_name": self.model_name,
                "graph_mode": self.graph_mode,
                "fold": fold_num,
                "state_dict": best_state,
                "model_kwargs": resolved_model_kwargs,
                "dataset_kwargs": ckpt_dataset_kwargs,
                "tabular_cols": self.tabular_cols if self.tabular_cols else None,
                "normalizer_state": normalizer.state_dict(),
                "reference_stats": reference_stats,
                "split": {
                    "n_splits": self.n_splits,
                    "seed": self.seed,
                    "train_indices": train_idx.tolist(),
                    "val_indices": val_idx.tolist(),
                    "train_subject_fingerprint": compute_split_fingerprint(df_train_raw["subject_id"].values),
                    "val_subject_fingerprint": compute_split_fingerprint(df_val_raw["subject_id"].values),
                },
                "metrics": best_metrics,
            }
            ckpt_path = os.path.join(
                self.out_dir, f"{self.model_name}_{self.graph_mode}_fold{fold_num}_best.pt"
            )
            torch.save(ckpt_bundle, ckpt_path)

            best_metrics["fold"] = fold_num
            fold_rows.append(best_metrics)
            all_y_true.extend(best_outputs[0].tolist())
            all_y_pred.extend(best_outputs[1].tolist())
            all_y_prob.append(best_outputs[2])

            print(
                f"  Fold {fold_num} Best | BalAcc: {best_metrics['balanced_accuracy']:.4f} | "
                f"Acc: {best_metrics['accuracy']:.4f} | F1: {best_metrics['f1_macro']:.4f} | "
                f"AUC: {best_metrics['auc']:.4f}"
            )

        folds_df = pd.DataFrame(fold_rows)
        folds_df.to_csv(os.path.join(self.out_dir, "fold_metrics.csv"), index=False)

        summary: Dict[str, float] = {}
        for col in folds_df.select_dtypes(include=[np.number]).columns:
            if col == "fold":
                continue
            summary[col] = float(folds_df[col].mean())
            summary[f"{col}_std"] = float(folds_df[col].std(ddof=0))

        save_json(os.path.join(self.out_dir, "summary_metrics.json"), summary)

        if self.save_plots and all_y_true:
            self._save_evaluation_plots(
                np.asarray(all_y_true),
                np.asarray(all_y_pred),
                np.concatenate(all_y_prob, axis=0),
            )

        print(
            f"\n✅ CV Complete | Mean BalAcc: {summary.get('balanced_accuracy', 0.0):.4f} ± "
            f"{summary.get('balanced_accuracy_std', 0.0):.4f} | "
            f"Mean AUC: {summary.get('auc', 0.0):.4f}"
        )
        return summary

    def test(self) -> Dict[str, float]:
        summary_path = os.path.join(self.out_dir, "summary_metrics.json")
        if os.path.exists(summary_path):
            import json

            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self.launch()

    def _save_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            import seaborn as sns

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        except ImportError:
            im = ax.imshow(cm, cmap="Blues")
            fig.colorbar(im, ax=ax)
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center")
        ax.set_title(f"{self.model_name} ({self.graph_mode}) Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "confusion_matrix.png"), dpi=200)
        plt.close(fig)

        if self.num_classes == 2 and y_prob.shape[1] >= 2 and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{self.model_name} ({self.graph_mode}) ROC Curve")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "roc_curve.png"), dpi=200)
            plt.close(fig)

