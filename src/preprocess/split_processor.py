from src.utils import ParameterKeys
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import os


class SplitProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        self._init_general()
        self._init_data()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, {})
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.classes = general_parameters.get(
            ParameterKeys.CLASSES, ["CN", "MCI", "AD"]
        )
        self.seed = general_parameters.get(ParameterKeys.SEED, 42)
        self.n_splits = general_parameters.get(ParameterKeys.N_SPLITS, 5)
        for i in range(self.n_splits):
            os.makedirs(f"{self.out_dir}/folds/{i}", exist_ok=True)
        self.test_size = general_parameters.get(ParameterKeys.TEST_SIZE, 10)

    def _init_data(
        self,
    ):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA, {})
        self.pth = dataset_parameters.get(ParameterKeys.DATA_ROOT)
        self.df = pd.read_csv(self.pth)
        self.patient_col = dataset_parameters.get(ParameterKeys.PATIENT)
        self.class_col = dataset_parameters.get(ParameterKeys.CLASSES)

    def launch(self):
        cons_dataset = self.df[self.df[self.class_col].isin(self.classes)].copy().reset_index(drop=True)
        # Remap labels to 0 .. C - 1 based on the ordering in self.classes
        class_to_label = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        cons_dataset["label"] = cons_dataset[self.class_col].map(class_to_label).astype(int)

        # 1. Split holdout
        train_val_idx, test_idx = next(
            StratifiedGroupKFold(n_splits=self.test_size, shuffle=True, random_state=self.seed).split(
                cons_dataset, cons_dataset[self.class_col], groups=cons_dataset[self.patient_col]
            )
        )
        test_df = cons_dataset.iloc[test_idx].reset_index(drop=True)
        train_val_df = cons_dataset.iloc[train_val_idx].reset_index(drop=True)

        # 2. Split train and val
        skfs = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for ix, (train_idx, test_idx) in enumerate(skfs.split(train_val_df, train_val_df[self.class_col], groups=train_val_df[self.patient_col])):
            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[test_idx].reset_index(drop=True)
            train_df.to_csv(f"{self.out_dir}/folds/{ix}/train.csv", index=False)
            val_df.to_csv(f"{self.out_dir}/folds/{ix}/val.csv", index=False)
        test_df.to_csv(f"{self.out_dir}/test.csv", index=False)



