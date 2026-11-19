import nibabel as nib
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from src.utils import ParameterKeys
import os
from copy import deepcopy
from src.data.utils import BatchKeys


class ROIFeatureProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        data_path = self.parameters.get(
            ParameterKeys.DATA,
            self.parameters.get(ParameterKeys.RAW_DATASET),
        )
        self.raw_dataset = pd.read_csv(data_path)
        self.out_dir = self.parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.atlas_path = self.parameters.get(ParameterKeys.ATLAS)
        self.pbar = self.parameters.get(ParameterKeys.PBAR, False)
        self.skip = self.parameters.get(ParameterKeys.SKIP, True)
        self.mri_dir = self.parameters.get(ParameterKeys.MRI_DIR)
        self.pet_dir = self.parameters.get(ParameterKeys.PET_DIR)
        self.reference_labels = self.parameters.get(ParameterKeys.LABEL)

    def get_pbar(
        self,
        data_list: list,
        desc: str = "",
    ):
        return (
            data_list
            if not self.pbar
            else tqdm(data_list, total=len(data_list), desc=desc)
        )

    def load_atlas(self):
        return nib.load(self.atlas_path).get_fdata()

    def compute_mri_features(self, mri_pth: str, atlas: np.ndarray) -> np.ndarray:
        mri = nib.load(f"{self.mri_dir}/{mri_pth}").get_fdata()
        atlas_tensor = np.rint(deepcopy(atlas)).astype(np.int16)
        if atlas_tensor.ndim == 4:
            atlas_tensor = atlas_tensor[..., 0]
        mri = np.asarray(mri, dtype=np.float32)
        if mri.ndim == 4:
            mri = mri[..., 0]
        if atlas_tensor.shape != mri.shape:
            raise ValueError(
                f"Shape mismatch: atlas {atlas_tensor.shape} vs. MRI {mri.shape}. "
                "Resample the atlas to match the MRI grid first."
            )
        # Map non-zero active AAL3 labels (1..170 excluding empty IDs) to contiguous indices 0..165
        roi_labels = sorted([int(x) for x in np.unique(atlas_tensor) if x > 0])
        num_rois = len(roi_labels)
        lut = np.zeros(int(atlas_tensor.max()) + 1, dtype=np.int16)
        for idx, label_id in enumerate(roi_labels):
            lut[label_id] = idx + 1
        fg_mask = atlas_tensor > 0
        roi_idx_fg = lut[atlas_tensor[fg_mask]] - 1
        roi_counts = np.maximum(
            np.bincount(roi_idx_fg, minlength=num_rois).astype(np.float32),
            1.0,
        )
        vals = mri[fg_mask]
        has_negative = bool(np.any(vals < 0.0))
        valid_vals = vals if has_negative else vals[vals > 0.0]
        if valid_vals.size < 100:
            return np.zeros((num_rois, 3), dtype=np.float32)
        # Clip extreme intensity outliers (0.5th - 99.5th percentile) and fit fast 1D 3-class K-Means
        p_low, p_high = np.percentile(valid_vals, [0.5, 99.5])
        clipped = np.clip(valid_vals, p_low, p_high)
        hist, bin_edges = np.histogram(clipped, bins=256)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_f = hist.astype(np.float64)
        centers = np.percentile(clipped, [15.0, 50.0, 85.0])
        for _ in range(30):
            distances = np.abs(bin_centers[:, None] - centers[None, :])
            assign = np.argmin(distances, axis=1)
            new_centers = np.array(
                [
                    np.sum(bin_centers[assign == k] * hist_f[assign == k])
                    / max(np.sum(hist_f[assign == k]), 1e-8)
                    for k in range(3)
                ]
            )
            if np.allclose(centers, new_centers, atol=1e-3):
                break
            centers = new_centers
        centers = np.sort(centers)
        t_csf_gm = 0.5 * (centers[0] + centers[1])
        t_gm_wm = 0.5 * (centers[1] + centers[2])
        fg_valid = np.ones_like(vals, dtype=bool) if has_negative else (vals > 0.0)
        csf_mask = fg_valid & (vals <= t_csf_gm)
        gm_mask = fg_valid & (vals > t_csf_gm) & (vals <= t_gm_wm)
        wm_mask = fg_valid & (vals > t_gm_wm)
        gm = np.bincount(
            roi_idx_fg, weights=gm_mask.astype(np.float32), minlength=num_rois
        )
        wm = np.bincount(
            roi_idx_fg, weights=wm_mask.astype(np.float32), minlength=num_rois
        )
        csf = np.bincount(
            roi_idx_fg, weights=csf_mask.astype(np.float32), minlength=num_rois
        )
        gm = gm / roi_counts
        wm = wm / roi_counts
        csf = csf / roi_counts
        return np.stack([gm, wm, csf], axis=1).astype(np.float32)

    def compute_pet_features(
        self, pet_pth: str, atlas: np.ndarray, reference_labels: list[int] = None, has_pet: bool = True,
    ) -> np.ndarray:
        atlas_tensor = np.rint(atlas).astype(np.int16)
        if atlas_tensor.ndim == 4:
            atlas_tensor = atlas_tensor[..., 0]
        roi_labels = sorted([int(x) for x in np.unique(atlas) if x > 0])
        num_rois = len(roi_labels)
        has_pet_flag = bool(has_pet) and not (isinstance(has_pet, str) and has_pet.lower() in ("false", "0", "none", "nan"))
        pet_file_valid = (
            has_pet_flag
            and pet_pth is not None
            and not pd.isna(pet_pth)
            and str(pet_pth).strip().lower() not in ("nan", "none", "")
            and os.path.exists(f"{self.pet_dir}/{pet_pth}")
        )
        if not pet_file_valid:
            return np.zeros((num_rois, 1), dtype=np.float32)
        pet_tensor = nib.load(f"{self.pet_dir}/{pet_pth}").get_fdata()
        pet = np.asarray(pet_tensor, dtype=np.float32)
        if pet.ndim == 4:
            pet = np.mean(pet, axis=-1)
        if atlas_tensor.shape != pet.shape:
            raise ValueError(
                f"Shape mismatch: atlas {atlas_tensor.shape} vs. PET {pet.shape}. "
                "Resample the atlas to match the PET grid first."
            )
        if reference_labels is None:
            # Cerebellum_Crus1_L (95) through Cerebellum_10_R (112), matching processing.yaml
            reference_labels = list(range(95, 113))
        lut = np.zeros(int(atlas_tensor.max()) + 1, dtype=np.int16)
        for idx, label_id in enumerate(roi_labels):
            lut[label_id] = idx + 1
        fg_mask = atlas_tensor > 0
        fg_atlas = atlas_tensor[fg_mask]
        fg_pet = pet[fg_mask]
        # Compute mean PET uptake inside the cerebellum reference region
        ref_mask = np.isin(fg_atlas, list(reference_labels))
        ref_vals = fg_pet[ref_mask]
        ref_valid = ref_vals[ref_vals > 0.0] if np.any(ref_vals > 0.0) else ref_vals
        ref_mean = float(np.mean(ref_valid)) if ref_valid.size > 0 else 0.0
        if ref_mean <= 1e-8:
            return np.zeros((num_rois, 1), dtype=np.float32)
        # Average only positive voxels per ROI so out-of-FOV zero padding does not dilute edge ROIs
        valid = fg_pet > 0.0
        if not np.any(valid):
            return np.zeros((num_rois, 1), dtype=np.float32)
        roi_idx_valid = lut[fg_atlas[valid]] - 1
        roi_sums = np.bincount(
            roi_idx_valid, weights=fg_pet[valid], minlength=num_rois
        ).astype(np.float32)
        roi_counts = np.bincount(roi_idx_valid, minlength=num_rois).astype(np.float32)
        roi_means = np.divide(
            roi_sums,
            roi_counts,
            out=np.zeros(num_rois, dtype=np.float32),
            where=roi_counts > 0,
        )
        suvr = (roi_means / ref_mean).astype(np.float32)
        return suvr[:, None]

    def launch(self):
        data_list = self.raw_dataset[
            ["mri_file", "pet_file", "safetensor_file", "has_pet"]
        ].values.tolist()
        bar = self.get_pbar(data_list)
        atlas = self.load_atlas()
        for item in bar:
            mri_file, pet_file, out_fname, has_pet = item
            if self.skip and os.path.exists(f"{self.out_dir}/{out_fname}"):
                continue
            mri_tensor = self.compute_mri_features(mri_pth=mri_file, atlas=atlas)
            pet_tensor = self.compute_pet_features(
                pet_pth=pet_file, atlas=atlas, reference_labels=self.reference_labels, has_pet=has_pet,
            )
            scan_tensor = torch.cat([
                torch.from_numpy(mri_tensor),
                torch.from_numpy(pet_tensor),
            ], dim=-1).float()
            save_file({BatchKeys.ROI: scan_tensor}, f"{self.out_dir}/{out_fname}")
