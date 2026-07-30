import pandas as pd
import numpy as np
from pathlib import Path

from config import FILE_ADNIMERGE, FILE_UPENN, OUTPUT_DIR

def clean_biomarker(val):
    """Clean strings like '>1700' or '<200' in biomarkers"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        if val.startswith('>'):
            return float(val.replace('>', ''))
        if val.startswith('<'):
            return float(val.replace('<', ''))
    return float(val)

def main():
    print("--- 1. Caricamento Dati ---")
    adni = pd.read_csv(FILE_ADNIMERGE, low_memory=False)
    upenn = pd.read_csv(FILE_UPENN, low_memory=False)
    
    print(f"ADNIMERGE shape: {adni.shape}")
    print(f"UPENN shape: {upenn.shape}")

    baseline_mci = adni[
        (adni['VISCODE'] == 'bl') & 
        (adni['DX_bl'].isin(['LMCI', 'EMCI', 'MCI']))
    ].copy()
    
    print(f"\nSoggetti MCI alla baseline: {len(baseline_mci)}")

    upenn_bl = upenn[upenn['VISCODE2'] == 'bl'].copy()
    
    for col in ['ABETA42', 'TAU', 'PTAU']:
        upenn_bl[col] = upenn_bl[col].apply(clean_biomarker)
    
    upenn_bl = upenn_bl.rename(columns={
        'ABETA42': 'ABETA_upenn', 
        'TAU': 'TAU_upenn', 
        'PTAU': 'PTAU_upenn'
    })
    
    merged = pd.merge(baseline_mci, upenn_bl[['RID', 'ABETA_upenn', 'TAU_upenn', 'PTAU_upenn']], 
                      on='RID', how='left')
    
    for bio in ['ABETA', 'TAU', 'PTAU']:
        merged[bio] = merged[bio].apply(clean_biomarker)
        
        upenn_col = f"{bio}_upenn" if bio != 'ABETA' else 'ABETA_upenn'
        
        merged[f'{bio}_final'] = merged[bio].fillna(merged[upenn_col])
        
    merged['CSF_Available'] = (~merged['ABETA_final'].isna()).astype(int)
    
    for bio in ['ABETA_final', 'TAU_final', 'PTAU_final']:
        merged[bio] = merged[bio].fillna(-1)

    print("\n--- Generazione Label (Follow-up) ---")
    rids = merged['RID'].unique()
    labels = {}
    
    history = adni[adni['RID'].isin(rids)][['RID', 'DX', 'Month']].copy()
    
    for rid in rids:
        subj_hist = history[history['RID'] == rid].sort_values('Month')
        
        diagnoses = subj_hist['DX'].unique()
        has_converted = any(d in ['Dementia', 'AD'] for d in diagnoses if pd.notnull(d))
        
        last_month = subj_hist['Month'].max()
        
        if has_converted:
            labels[rid] = 1
        elif last_month >= 12:
            labels[rid] = 0
        else:
            labels[rid] = -1
            
    merged['label'] = merged['RID'].map(labels)
    
    final_df = merged[merged['label'] != -1].copy()
    
    cols_to_keep = [
        'RID', 'PTID', 'label', 
        'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE',
        'ABETA_final', 'TAU_final', 'PTAU_final', 'CSF_Available'
    ]
    
    final_df['Gender_Male'] = (final_df['PTGENDER'] == 'Male').astype(int)
    final_df['APOE4'] = final_df['APOE4'].fillna(0)
    final_df['MMSE'] = final_df['MMSE'].fillna(final_df['MMSE'].mean())
    
    final_dataset = final_df[cols_to_keep + ['Gender_Male']].copy()
    
    print(f"\nDataset Finale Generato: {len(final_dataset)} pazienti.")
    print(f"Distribuzione Classi: {final_dataset['label'].value_counts().to_dict()}")
    print(f"Dataset salvato in: {OUTPUT_DIR / 'clinical_features.csv'}")
    
    final_dataset.to_csv(OUTPUT_DIR / 'clinical_features.csv', index=False)

if __name__ == "__main__":
    main()