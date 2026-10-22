from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from src.utils.general import resolve_data_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATLAS_PATH = PROJECT_ROOT / "configs" / "AAL3" / "AAL3v1.nii"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "configs" / "aal3_coordinates.csv"


def extract_aal3_coordinates(
    atlas_path: Path = DEFAULT_ATLAS_PATH,
    output_file: Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    print("--- Extraction of AAL3 Centroid Coordinates ---")
    atlas_path = Path(resolve_data_path(atlas_path))
    output_file = Path(resolve_data_path(output_file))

    if not atlas_path.exists():
        gz_candidate = atlas_path.with_suffix(".nii.gz")
        if gz_candidate.exists():
            atlas_path = gz_candidate
        else:
            raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    print(f"Using atlas: {atlas_path}")
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata()

    unique_labels = sorted([int(i) for i in np.unique(atlas_data) if i != 0])
    print(f"Found {len(unique_labels)} unique regions in NIfTI file.")

    centroids = []
    affine = atlas_img.affine

    for label_id in unique_labels:
        coords_voxel = np.argwhere(atlas_data == label_id)
        if len(coords_voxel) == 0:
            continue
        center_voxel = coords_voxel.mean(axis=0)
        center_mni = nib.affines.apply_affine(affine, center_voxel)
        centroids.append(
            {
                "label_id": label_id,
                "x": float(center_mni[0]),
                "y": float(center_mni[1]),
                "z": float(center_mni[2]),
            }
        )

    df_coords = pd.DataFrame(centroids).sort_values("label_id")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_coords.to_csv(output_file, index=False)
    print(f"\nCoordinates saved in: {output_file} (shape={df_coords.shape})")
    return df_coords


class AtlasCoordsExtractor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        atlas_cfg = parameters.get("atlas", parameters)
        self.atlas_path = Path(atlas_cfg.get("aal_file", DEFAULT_ATLAS_PATH))
        self.output_file = Path(atlas_cfg.get("output_file", DEFAULT_OUTPUT_FILE))

    def launch(self):
        return extract_aal3_coordinates(self.atlas_path, self.output_file)

    def __call__(self):
        return self.launch()


if __name__ == "__main__":
    extract_aal3_coordinates()
