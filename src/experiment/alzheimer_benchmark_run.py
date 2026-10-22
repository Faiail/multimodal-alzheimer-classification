import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.data.preprocessing import FoldFeatureNormalizer, imaging_feature_columns
from src.experiment.alzheimer_gnn_run import AlzheimerGNNRun, prepare_alzheimer_dataframe
from src.metrics import compute_classification_metrics
from src.utils import ParameterKeys, save_json, set_seed


class AlzheimerBenchmarkRun:
    """Multi-model and multi-topology benchmark runner (XGBoost, GCN, GAT, DualGraph)."""

    def __init__(self, parameters: dict):
        self.parameters = deepcopy(parameters)
        self.general = self.parameters.get(ParameterKeys.GENERAL, {})
        self.out_dir = self.general.get(ParameterKeys.OUT_DIR, "models/benchmark")
        os.makedirs(self.out_dir, exist_ok=True)
        self.seed = int(self.general.get(ParameterKeys.SEED, 42))
        self.n_splits = int(
            self.general.get(
                ParameterKeys.N_SPLITS,
                self.parameters.get("training", {}).get("n_splits", 5),
            )
        )
        set_seed(self.seed)

        bench_cfg = self.parameters.get("benchmark", {})
        self.models: List[str] = list(
            bench_cfg.get("models", ["XGBoost", "GCN_KAN_Classifier", "GAT_KAN_Classifier", "DualGraph_KAN_Classifier"])
        )
        self.graph_modes: List[str] = list(
            bench_cfg.get("graph_modes", ["static", "zhang", "hybrid", "knn", "multigraph"])
        )

    def _run_xgboost_cv(self) -> Dict[str, float]:
        import xgboost as xgb

        data_cfg = self.parameters.get(ParameterKeys.DATA, {})
        df_full = prepare_alzheimer_dataframe(data_cfg)
        target_col = data_cfg.get("target_col", "target")
        num_classes = int(
            self.parameters.get(ParameterKeys.MODEL, {})
            .get(ParameterKeys.CFG, self.parameters.get(ParameterKeys.MODEL, {}))
            .get("out_classes", 3)
        )

        if "subject_id" not in df_full.columns:
            df_full["subject_id"] = (
                df_full["PTID"] if "PTID" in df_full.columns else [f"subj_{i}" for i in range(len(df_full))]
            )

        feat_cols = imaging_feature_columns(df_full)
        tab_cols = list(data_cfg.get("tabular_cols", []))
        all_cols = feat_cols + [c for c in tab_cols if c in df_full.columns]

        y_all = df_full[target_col].values
        groups = df_full["subject_id"].values
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        fold_rows = []
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df_full, y_all, groups=groups)):
            df_train_raw = df_full.iloc[train_idx].reset_index(drop=True)
            df_val_raw = df_full.iloc[val_idx].reset_index(drop=True)

            normalizer = FoldFeatureNormalizer().fit(df_train_raw)
            df_train = normalizer.transform(df_train_raw)
            df_val = normalizer.transform(df_val_raw)

            X_train = df_train[all_cols].fillna(0).values
            y_train = df_train[target_col].values
            X_val = df_val[all_cols].fillna(0).values
            y_val = df_val[target_col].values

            sample_weights = compute_sample_weight("balanced", y_train)
            objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
            xgb_kwargs = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": objective,
                "random_state": self.seed + fold_idx,
                "n_jobs": 4,
            }
            if num_classes > 2:
                xgb_kwargs["num_class"] = num_classes

            clf = xgb.XGBClassifier(**xgb_kwargs)
            clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            y_prob = clf.predict_proba(X_val)
            y_pred = np.argmax(y_prob, axis=1)

            metrics = compute_classification_metrics(
                y_true=y_val,
                y_pred=y_pred,
                y_prob=y_prob,
                num_classes=num_classes,
            )
            metrics["fold"] = fold_idx + 1
            fold_rows.append(metrics)

        folds_df = pd.DataFrame(fold_rows)
        xgb_dir = os.path.join(self.out_dir, "XGBoost_tabular")
        os.makedirs(xgb_dir, exist_ok=True)
        folds_df.to_csv(os.path.join(xgb_dir, "fold_metrics.csv"), index=False)

        summary = {
            "model": "XGBoost",
            "graph_mode": "tabular",
        }
        for col in folds_df.select_dtypes(include=[np.number]).columns:
            if col == "fold":
                continue
            summary[col] = float(folds_df[col].mean())
            summary[f"{col}_std"] = float(folds_df[col].std(ddof=0))
        save_json(os.path.join(xgb_dir, "summary_metrics.json"), summary)
        return summary

    def launch(self):
        summary_rows = []
        for model_name in self.models:
            if model_name.lower() == "xgboost":
                print("\n=== Running XGBoost Baseline ===")
                summary_rows.append(self._run_xgboost_cv())
                continue

            for graph_mode in self.graph_modes:
                print(f"\n=== Running {model_name} | graph_mode={graph_mode} ===")
                run_params = deepcopy(self.parameters)
                run_params.setdefault(ParameterKeys.GENERAL, {})[
                    ParameterKeys.OUT_DIR
                ] = os.path.join(self.out_dir, f"{model_name}_{graph_mode}")
                run_params.setdefault(ParameterKeys.DATA, {})["graph_mode"] = graph_mode
                run_params.setdefault(ParameterKeys.MODEL, {})[ParameterKeys.NAME] = model_name

                runner = AlzheimerGNNRun(run_params)
                res = runner.launch()
                res["model"] = model_name
                res["graph_mode"] = graph_mode
                summary_rows.append(res)

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(self.out_dir, "benchmark_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n✅ Saved benchmark summary to: {summary_csv}")
        return summary_rows

    def test(self):
        return self.launch()
