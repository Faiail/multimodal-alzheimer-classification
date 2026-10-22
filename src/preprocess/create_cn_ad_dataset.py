import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocess.config import FILE_ADNIMERGE, OUTPUT_DIR
from src.utils.general import resolve_data_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = OUTPUT_DIR

def create_cn_ad_dataset(
    adni_path=FILE_ADNIMERGE,
    output_file=OUTPUT_DIR / "cn_ad_features.csv",
):
    """
    Creates CN vs AD dataset with clinical features:
    - 0: CN (Cognitively Normal)
    - 1: AD (Alzheimer's Disease)
    """
    print("🔄 Creating CN vs AD dataset...")

    adni_path = Path(resolve_data_path(adni_path))
    if not adni_path.exists():
        raise FileNotFoundError(f"File ADNIMERGE non trovato: {adni_path}")

    adni = pd.read_csv(adni_path, low_memory=False)
    print(f"📂 Caricato ADNIMERGE: {len(adni)} righe")

    baseline_patients = adni[
        (adni['VISCODE'] == 'bl') &
        (adni['DX_bl'].isin(['CN', 'AD']))
    ].copy()

    print(f"✅ Pazienti baseline CN/AD: {len(baseline_patients)}")
    print(f"   📊 Distribuzione: {baseline_patients['DX_bl'].value_counts().to_dict()}")

    clinical_cols = [
        'RID', 'PTID', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE',
        'ABETA', 'TAU', 'PTAU', 'DX_bl'
    ]

    missing_cols = [col for col in clinical_cols if col not in baseline_patients.columns]
    if missing_cols:
        print(f"⚠️ Colonne mancanti in ADNIMERGE: {missing_cols}")
        clinical_cols = [col for col in clinical_cols if col in baseline_patients.columns]

    df_clinical = baseline_patients[clinical_cols].copy()

    def clean_biomarker(val):
        """Pulisce biomarker con valori tipo '>1700' o '<200'"""
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('>'):
                return float(val[1:])
            elif val.startswith('<'):
                return float(val[1:])
        try:
            return float(val)
        except:
            return np.nan

    for bio in ['ABETA', 'TAU', 'PTAU']:
        if bio in df_clinical.columns:
            df_clinical[f'{bio}_final'] = df_clinical[bio].apply(clean_biomarker)
        else:
            df_clinical[f'{bio}_final'] = np.nan

    df_clinical['CSF_Available'] = (~df_clinical['ABETA_final'].isna()).astype(int)

    for bio in ['ABETA_final', 'TAU_final', 'PTAU_final']:
        df_clinical[bio] = df_clinical[bio].fillna(-1)

    df_clinical['Gender_Male'] = (df_clinical['PTGENDER'] == 'Male').astype(int)

    df_clinical['label'] = (df_clinical['DX_bl'] == 'AD').astype(int)

    final_cols = ['RID', 'PTID', 'label', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE',
                  'ABETA_final', 'TAU_final', 'PTAU_final', 'CSF_Available', 'Gender_Male']

    df_final = df_clinical[final_cols].copy()

    print("📊 CN vs AD dataset created:")
    print(f"   📈 Total patients: {len(df_final)}")
    print(f"   🎯 Class distribution:")
    for label, count in df_final['label'].value_counts().sort_index().items():
        class_name = 'CN' if label == 0 else 'AD'
        percentage = count / len(df_final) * 100
        print(f"      {class_name}: {count} ({percentage:.1f}%)")
    output_file = Path(resolve_data_path(output_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"\nDataset saved: {output_file}")

    print("📋 Summary:")
    print(f"   • CN (label 0): {len(df_final[df_final['label'] == 0])} patients")
    print(f"   • AD (label 1): {len(df_final[df_final['label'] == 1])} patients")
    print(f"   • Columns: {len(df_final.columns)}")

    return df_final


class CNADDatasetBuilder:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        cfg = parameters.get("preprocess", parameters)
        self.adni_path = cfg.get("adnimerge_path", FILE_ADNIMERGE)
        self.output_file = cfg.get("output_file", OUTPUT_DIR / "cn_ad_features.csv")

    def launch(self):
        return create_cn_ad_dataset(self.adni_path, self.output_file)

    def __call__(self):
        return self.launch()


if __name__ == "__main__":
    create_cn_ad_dataset()