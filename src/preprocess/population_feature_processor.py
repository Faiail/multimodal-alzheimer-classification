import torch
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import ParameterKeys
from safetensors.torch import load_file
from src.data.utils import BatchKeys
from einops import rearrange
import joblib


class PopulationFeatureProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        self.fold_dir = self.parameters.get(ParameterKeys.FOLD)
        self.dataset = pd.read_csv(self.parameters.get(ParameterKeys.DATA))
        self.test_set = pd.read_csv(self.parameters.get(ParameterKeys.TEST))
        self.feature_dir = self.parameters.get(ParameterKeys.FEATURES)
        self.out_dir = self.parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.seed = self.parameters.get(ParameterKeys.SEED)

    def get_features(self, files: list[str]) -> np.ndarray:
        stacked = torch.stack(
            [load_file(f"{self.feature_dir}/{x}")[BatchKeys.ROI] for x in files]
        )
        return rearrange(stacked, "b r f -> b (r f)").numpy()

    def launch(self):
        # instanciating pipeline
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "rfe",
                    RFE(
                        estimator=RidgeClassifier(alpha=1.0, random_state=self.seed),
                        n_features_to_select=128,
                        step=0.1,
                    ),
                ),
            ]
        )

        # getting train df
        train_df = pd.read_csv(f"{self.fold_dir}/train.csv")
        X_train = self.get_features(train_df["safetensor_file"])
        y_train = train_df["label"]

        # train feature selector
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f"{self.out_dir}/pipeline.pkl")

        dataset_sorted = self.dataset.sort_values("node_id", ascending=True).reset_index(drop=True)
        all_features = self.get_features(dataset_sorted["safetensor_file"])
        feature_selected = pipeline.transform(all_features)
        torch.save(feature_selected, f"{self.out_dir}/features.pt")

