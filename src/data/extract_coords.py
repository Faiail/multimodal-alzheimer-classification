import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting
from pathlib import Path
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'processing_config.yaml'
OUTPUT_DIR = PROJECT_ROOT / 'data' / '03_features'

def main():
    print("--- Extraction of AAL3 Centroid Coordinates (Consistent with Feature Extraction) ---")
    
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found in: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    
    atlas_path_str = config.get('atlas', {}).get('aal_file')
    if not atlas_path_str:
        atlas_path = PROJECT_ROOT / 'configs' / 'AAL3' / 'AAL3v1.nii'
    else:
        atlas_path = Path(atlas_path_str)
        
    print(f"Using atlas: {atlas_path}")
    
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata()
    
    unique_labels = sorted([int(i) for i in np.unique(atlas_data) if i != 0])
    print(f"Found {len(unique_labels)} unique regions in NIfTI file.")
    
    centroids = []
    
    affine = atlas_img.affine
    
    print("Calculating centroids...")
    for label_id in unique_labels:
        coords_voxel = np.argwhere(atlas_data == label_id)
        
        if len(coords_voxel) == 0:
            print(f"Warning: Label {label_id} exists in unique but has no voxels?!")
            continue
            
        center_voxel = coords_voxel.mean(axis=0)
        
        center_mni = nib.affines.apply_affine(affine, center_voxel)
        
        centroids.append({
            'label_id': label_id,
            'x': center_mni[0],
            'y': center_mni[1],
            'z': center_mni[2]
        })

    df_coords = pd.DataFrame(centroids)
    
    df_coords = df_coords.sort_values('label_id')
    
    output_file = OUTPUT_DIR / "aal3_coordinates.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df_coords.to_csv(output_file, index=False)
    print(f"\nCoordinates saved in: {output_file}")
    print(f"Final shape: {df_coords.shape}")
    print("First 5 rows:")
    print(df_coords.head())

if __name__ == "__main__":
    main()