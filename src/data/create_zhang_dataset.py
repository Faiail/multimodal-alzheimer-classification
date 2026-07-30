import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gzip
import gc

try:
    from config import FILE_FEATURES, FILE_ADNIMERGE, OUTPUT_DIR
except ImportError:
    RAW_DATA_DIR = Path("data/raw")
    OUTPUT_DIR = Path("data/processed")
    FILE_ADNIMERGE = "ADNIMERGE_31Dec2025.csv"
    FILE_FEATURES = "data/04_final/features_matrix.csv"

USE_ONLY_GM = False 

def compute_ibn_vector(patient_row, nc_mean, nc_std, roi_cols):
    """Computes the flattened vector (Upper Triangle) with strict Type-Safety"""
    try:
        f_i = np.asarray(patient_row[roi_cols].values, dtype=np.float32)
        f_nc = np.asarray(nc_mean, dtype=np.float32)
        s_nc = np.asarray(nc_std, dtype=np.float32)
        
        n_rois = len(roi_cols)
        
        delta = f_i - f_nc
        
        num_mat = np.abs(np.subtract.outer(delta, delta))
        
        var_nc = s_nc ** 2
        denom_mat = np.add.outer(var_nc, var_nc) / 2.0
        denom_mat = np.sqrt(denom_mat) + 1e-9
        
        E_mat = num_mat / denom_mat
        
        if E_mat.dtype == 'O':
            E_mat = E_mat.astype(np.float32)
            
        R_mat = np.tanh(E_mat)
        
        del num_mat, denom_mat, E_mat, delta
        
        upper_tri_indices = np.triu_indices(n_rois, k=1)
        ibn_vector = R_mat[upper_tri_indices]
        
        return ibn_vector
    except Exception as e:
        return None

def main():
    print("--- ZHANG ET AL. FEATURE ENGINEERING (TYPE-SAFE MODE) ---")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df_feats = pd.read_csv(FILE_FEATURES)
    adni = pd.read_csv(FILE_ADNIMERGE, low_memory=False)
    
    df_feats['PTID'] = df_feats['subject_id'].apply(lambda x: "_".join(x.split('_')[:3]))
    
    df_merged = pd.merge(df_feats, adni[['PTID', 'DX_bl']], on='PTID', how='left')
    df_merged = df_merged.drop_duplicates(subset=['subject_id'])
    
    gm_cols = [c for c in df_feats.columns if c.startswith('GM_vol_Region_')]
    pet_cols = [c for c in df_feats.columns if c.startswith('PET_suvr_Region_')]
    
    if USE_ONLY_GM:
        roi_cols = gm_cols
        print(f"⚠️ REDUCED MODE: Using only {len(roi_cols)} ROI (GM Only)")
    else:
        roi_cols = gm_cols + pet_cols
        print(f"✅ COMPLETE MODE: Using {len(roi_cols)} ROI (GM + PET)")

    if len(roi_cols) == 0:
        raise ValueError("ERROR: No ROI columns found.")

    print("Pre-processing NaN and converting to float32 globally...")
    df_merged[roi_cols] = df_merged[roi_cols].fillna(0).astype(np.float32)
    
    nc_subjects = df_merged[df_merged['DX_bl'] == 'CN']
    print(f"Normal Control subjects for reference: {len(nc_subjects)}")

    if len(nc_subjects) < 5:
        print("WARNING: Few CN subjects. Using global statistics.")
        nc_mean = df_merged[roi_cols].mean().values
        nc_std = df_merged[roi_cols].std().values
    else:
        nc_mean = nc_subjects[roi_cols].mean().values
        nc_std = nc_subjects[roi_cols].std().values
        
    nc_std[nc_std == 0] = 1.0

    out_path = Path(OUTPUT_DIR) / 'zhang_features.csv.gz'
    
    n_rois = len(roi_cols)
    n_features = int(n_rois * (n_rois - 1) / 2)
    print(f"Generating {n_features} features per patient (Input ROIs: {n_rois})...")

    with gzip.open(out_path, 'wt') as f:
        header = "subject_id," + ",".join([f"IBN_{i}" for i in range(n_features)]) + "\n"
        f.write(header)
        
        count = 0
        skipped = 0
        
        for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged)):
            
            subj_id = str(row['subject_id']).replace(',', '_').strip()
            
            ibn_vec = compute_ibn_vector(row, nc_mean, nc_std, roi_cols)
            
            if ibn_vec is None:
                skipped += 1
                continue
            
            if len(ibn_vec) != n_features:
                print(f"Vector length error for {subj_id}")
                skipped += 1
                continue
            
            vec_str = ",".join((f"{x:.4f}" for x in ibn_vec))
            line = f"{subj_id},{vec_str}\n"
            f.write(line)
            
            del ibn_vec, line, vec_str
            count += 1
            if count % 50 == 0: gc.collect()

    print(f"Done! Dataset saved in: {out_path}")
    print(f"Processed patients: {count}, Skipped: {skipped}")

if __name__ == "__main__":
    main()