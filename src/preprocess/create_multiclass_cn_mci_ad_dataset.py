import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocess.config import OUTPUT_DIR
from src.utils.general import resolve_data_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = OUTPUT_DIR

def create_multiclass_dataset(
    mci_file=DATA_DIR / "clinical_features.csv",
    cn_ad_file=DATA_DIR / "cn_ad_features.csv",
    output_file=OUTPUT_DIR / "multiclass_cn_mci_ad_features.csv",
):
    """
    Creates CN vs MCI vs AD multiclass dataset:
    - 0: CN (Cognitively Normal)
    - 1: MCI (Mild Cognitive Impairment - sMCI + pMCI)
    - 2: AD (Alzheimer's Disease)
    """
    print("🔄 Creating CN vs MCI vs AD multiclass dataset...")

    print("📂 Loading clinical_features.csv (MCI)...")
    mci_df = pd.read_csv(resolve_data_path(mci_file))
    print(f"   ✅ MCI: {len(mci_df)} pazienti")
    print(f"   📊 Distribuzione MCI: {mci_df['label'].value_counts().to_dict()}")

    mci_df['label'] = 1
    mci_df['original_class'] = 'MCI'

    print("\n📂 Loading cn_ad_features.csv (CN + AD)...")
    cn_ad_df = pd.read_csv(resolve_data_path(cn_ad_file))
    print(f"   ✅ CN+AD: {len(cn_ad_df)} pazienti")
    print(f"   📊 Distribuzione originale: {cn_ad_df['label'].value_counts().to_dict()}")

    cn_ad_df['label'] = cn_ad_df['label'].map({0: 0, 1: 2})
    cn_ad_df['original_class'] = cn_ad_df['label'].map({0: 'CN', 2: 'AD'})

    print("\n🔗 Combining datasets...")
    multiclass_df = pd.concat([mci_df, cn_ad_df], ignore_index=True)

    duplicates = multiclass_df['PTID'].duplicated().sum()
    if duplicates > 0:
        print(f"   ⚠️  Found {duplicates} duplicate PTIDs - will be removed")
        multiclass_df = multiclass_df.drop_duplicates(subset='PTID', keep='first')

    print("\n📊 Multiclass dataset created:")
    print(f"   📈 Total patients: {len(multiclass_df)}")
    print(f"   🎯 Class distribution:")
    for label, count in multiclass_df['label'].value_counts().sort_index().items():
        class_name = {0: 'CN', 1: 'MCI', 2: 'AD'}[label]
        percentage = count / len(multiclass_df) * 100
        print(f"      {class_name} (label {label}): {count} pazienti ({percentage:.1f}%)")
    expected_cols = ['RID', 'PTID', 'label', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE',
                     'ABETA_final', 'TAU_final', 'PTAU_final', 'CSF_Available', 'Gender_Male', 'original_class']
    missing_cols = set(expected_cols) - set(multiclass_df.columns)
    if missing_cols:
        print(f"   ⚠️  Missing columns: {missing_cols}")

    multiclass_df = multiclass_df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_file = Path(resolve_data_path(output_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    multiclass_df.to_csv(output_file, index=False)
    print(f"\nDataset saved: {output_file}")

    print("\n📋 Summary:")
    print(f"   • CN (label 0): {len(multiclass_df[multiclass_df['label'] == 0])} patients")
    print(f"   • MCI (label 1): {len(multiclass_df[multiclass_df['label'] == 1])} patients")
    print(f"   • AD (label 2): {len(multiclass_df[multiclass_df['label'] == 2])} patients")
    print(f"   • Columns: {len(multiclass_df.columns)}")
    print(f"   • Unique PTIDs: {multiclass_df['PTID'].nunique()}")

    return multiclass_df


class MulticlassDatasetBuilder:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        cfg = parameters.get("preprocess", parameters)
        self.mci_file = cfg.get("mci_file", DATA_DIR / "clinical_features.csv")
        self.cn_ad_file = cfg.get("cn_ad_file", DATA_DIR / "cn_ad_features.csv")
        self.output_file = cfg.get("output_file", OUTPUT_DIR / "multiclass_cn_mci_ad_features.csv")

    def launch(self):
        return create_multiclass_dataset(self.mci_file, self.cn_ad_file, self.output_file)

    def __call__(self):
        return self.launch()


if __name__ == "__main__":
    create_multiclass_dataset()