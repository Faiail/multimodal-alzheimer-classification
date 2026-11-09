from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    NormalizeIntensity,
    Resize,
)
import nibabel as nib
import numpy as np
import pandas as pd
from safetensors.torch import save_file
import torch
from tqdm import tqdm

from src.utils import ParameterKeys


class MultimodalTensorProcessor:
    """
    Multimodal 3D Tensorial Preprocessing with MONAI and SafeTensors.

    For each row in the dataset:
      - Discovers and matches co-registered sMRI (MPRAGE) and PET volumes in standard atlas space.
      - Applies MONAI 3D spatial resizing and intensity normalization.
      - Serializes the co-registered MRI-PET couple into a single .safetensors file named
        <patient_id><scan_id>.safetensors.
      - Modifies/augments the dataset CSV with columns for modality availability:
        'has_mri', 'has_pet', 'has_both', 'safetensor_file'.
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        # 1. Paths configuration
        raw_dataset_path = self.parameters.get(
            ParameterKeys.RAW_DATASET,
            self.parameters.get(ParameterKeys.DATA_ROOT, "data/processed/all_features.csv"),
        )
        self.raw_dataset_path = Path(raw_dataset_path)

        mri_dir = self.parameters.get(
            ParameterKeys.MRI_DIR, "data/processed/mri_atlas_registered"
        )
        self.mri_dir = Path(mri_dir)

        pet_dir = self.parameters.get(
            ParameterKeys.PET_DIR, "data/processed/pet_atlas_registered"
        )
        self.pet_dir = Path(pet_dir)

        out_dir = self.parameters.get(
            ParameterKeys.OUT_DIR, "data/processed/safetensors"
        )
        self.output_dir = Path(out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        out_csv = self.parameters.get(
            ParameterKeys.OUT_CSV,
            str(self.raw_dataset_path.parent / f"{self.raw_dataset_path.stem}_multimodal.csv"),
        )
        self.output_csv = Path(out_csv)

        # 2. Processing parameters
        self.skip_existing = self.parameters.get(ParameterKeys.SKIP, True)
        self.target_size = tuple(self.parameters.get(ParameterKeys.TARGET_SIZE, (128, 128, 128)))
        self.normalize_pet = self.parameters.get(ParameterKeys.NORMALIZE_PET, True)
        self.only_paired = self.parameters.get(ParameterKeys.ONLY_PAIRED, False)

        # 3. MONAI Transforms
        self.mri_transform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(spatial_size=self.target_size, mode="trilinear"),
            NormalizeIntensity(nonzero=True, channel_wise=True),  # Brain Z-score
            EnsureType(data_type="tensor", dtype=torch.float32),
        ])

        pet_transforms = [
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(spatial_size=self.target_size, mode="trilinear"),
        ]
        if self.normalize_pet:
            pet_transforms.append(NormalizeIntensity(nonzero=True, channel_wise=True))
        pet_transforms.append(EnsureType(data_type="tensor", dtype=torch.float32))

        self.pet_transform = Compose(pet_transforms)

    def _extract_id_tokens(self, row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts patient_id (e.g. '012_S_4849') and scan_id (e.g. 'I361978') from a row.
        Checks 'subject_id', 'PTID', 'scan_id', 'IMAGEUID', or index tokens.
        """
        raw_val = None
        for col in ["subject_id", "subject", "image_id", "filename", "ID"]:
            if col in row and pd.notna(row[col]):
                raw_val = str(row[col])
                break

        patient_id = None
        scan_id = None

        if raw_val:
            # Match ADNI PTID pattern xxx_S_xxxx
            m_ptid = re.search(r"([0-9]{3}_S_[0-9]{4})", raw_val)
            if m_ptid:
                patient_id = m_ptid.group(1)

            # Match Scan ID (last token I<digits>)
            m_scan = re.search(r"_(I[0-9]+)(?:_|\.|$)", raw_val)
            if m_scan:
                scan_id = m_scan.group(1)

        # Fallback to explicit columns if not extracted from subject_id
        if not patient_id and "PTID" in row and pd.notna(row["PTID"]):
            patient_id = str(row["PTID"]).strip()

        if not scan_id:
            for col in ["scan_id", "IMAGEUID", "IMAGEUID_bl", "image_uid"]:
                if col in row and pd.notna(row[col]):
                    val_str = str(row[col]).strip()
                    try:
                        val_float = float(val_str)
                        scan_id = f"I{int(val_float)}"
                    except ValueError:
                        scan_id = val_str if val_str.startswith("I") else f"I{val_str}"
                    break

        return patient_id, scan_id

    def _index_imaging_files(self) -> Tuple[Dict[str, Path], Dict[str, List[Path]], Dict[str, Path], Dict[str, List[Path]]]:
        """
        Indexes preprocessed MNI sMRI and PET files by scan_id and by patient_id.
        """
        mri_by_scan: Dict[str, Path] = {}
        mri_by_patient: Dict[str, List[Path]] = {}
        for mri_file in sorted(self.mri_dir.glob("*_mni.nii*")):
            clean = mri_file.name.replace("_mni.nii.gz", "").replace("_mni.nii", "")
            m_ptid = re.search(r"([0-9]{3}_S_[0-9]{4})", clean)
            m_scan = re.search(r"_(I[0-9]+)(?:_|$)", clean)
            if m_ptid:
                ptid = m_ptid.group(1)
                mri_by_patient.setdefault(ptid, []).append(mri_file)
            if m_scan:
                scan = m_scan.group(1)
                mri_by_scan[scan] = mri_file

        pet_by_scan: Dict[str, Path] = {}
        pet_by_patient: Dict[str, List[Path]] = {}
        for pet_file in sorted(self.pet_dir.glob("*_mni.nii*")):
            clean = pet_file.name.replace("_coreg_mni.nii.gz", "").replace("_coreg_mni.nii", "")
            m_ptid = re.search(r"([0-9]{3}_S_[0-9]{4})", clean)
            m_scan = re.search(r"_(I[0-9]+)(?:_|$)", clean)
            if m_ptid:
                ptid = m_ptid.group(1)
                pet_by_patient.setdefault(ptid, []).append(pet_file)
            if m_scan:
                scan = m_scan.group(1)
                pet_by_scan[scan] = pet_file

        return mri_by_scan, mri_by_patient, pet_by_scan, pet_by_patient

    def _load_nifti_data(self, path: Path) -> np.ndarray:
        """Loads NIfTI volume, converting multi-frame 4D to 3D average if needed."""
        img = nib.load(str(path))
        data = img.get_fdata().astype(np.float32)
        if data.ndim == 4 and data.shape[-1] > 1:
            data = np.mean(data, axis=-1).astype(np.float32)
        elif data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        return data

    def launch(self):
        if not self.raw_dataset_path.exists():
            raise FileNotFoundError(f"Raw dataset CSV not found at: {self.raw_dataset_path}")

        print(f"[Multimodal Tensor Processor] Starting tensorial pipeline...")
        print(f"  • Raw dataset:  {self.raw_dataset_path}")
        print(f"  • MRI atlas:    {self.mri_dir}")
        print(f"  • PET atlas:    {self.pet_dir}")
        print(f"  • Output dir:   {self.output_dir}")
        print(f"  • Target size:  {self.target_size}")
        print(f"  • Normalize PET:{self.normalize_pet}")

        # Index available preprocessed files
        mri_by_scan, mri_by_patient, pet_by_scan, pet_by_patient = self._index_imaging_files()
        print(f"  • Found {len(mri_by_scan)} MRI scans ({len(mri_by_patient)} unique patients)")
        print(f"  • Found {len(pet_by_scan)} PET scans ({len(pet_by_patient)} unique patients)")

        df = pd.read_csv(self.raw_dataset_path, low_memory=False)
        print(f"  • Processing {len(df)} rows from dataset...")

        has_mri_list = []
        has_pet_list = []
        has_both_list = []
        safetensor_file_list = []

        count_created = 0
        count_skipped = 0
        count_no_files = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating SafeTensors"):
            ptid, scan_id = self._extract_id_tokens(row)

            # Match MRI file
            mri_path: Optional[Path] = None
            if scan_id and scan_id in mri_by_scan:
                mri_path = mri_by_scan[scan_id]
            elif ptid and ptid in mri_by_patient:
                mri_path = mri_by_patient[ptid][0]

            # Match PET file (co-registered to patient's sMRI)
            pet_path: Optional[Path] = None
            if scan_id and scan_id in pet_by_scan:
                pet_path = pet_by_scan[scan_id]
            elif ptid and ptid in pet_by_patient:
                pet_path = pet_by_patient[ptid][0]

            has_mri = mri_path is not None
            has_pet = pet_path is not None
            has_both = has_mri and has_pet

            has_mri_list.append(has_mri)
            has_pet_list.append(has_pet)
            has_both_list.append(has_both)

            if (self.only_paired and not has_both) or (not has_mri and not has_pet):
                safetensor_file_list.append("")
                count_no_files += 1
                continue

            # Naming format: <patient_id><scan_id>.safetensors
            fname_ptid = ptid if ptid else "UNKNOWN"
            fname_scan = scan_id if scan_id else "I0"
            st_filename = f"{fname_ptid}{fname_scan}.safetensors"
            st_path = self.output_dir / st_filename
            safetensor_file_list.append(st_filename)

            if self.skip_existing and st_path.exists():
                count_skipped += 1
                continue

            tensors_to_save = {}

            if has_mri:
                mri_raw = self._load_nifti_data(mri_path)
                mri_tensor = self.mri_transform(mri_raw)
                tensors_to_save["mri"] = mri_tensor

            if has_pet:
                pet_raw = self._load_nifti_data(pet_path)
                pet_tensor = self.pet_transform(pet_raw)
                tensors_to_save["pet"] = pet_tensor

            save_file(tensors_to_save, str(st_path))
            count_created += 1

        # Augment dataframe with presence columns without fragmentation
        df = df.copy()
        extra_df = pd.DataFrame({
            "has_mri": has_mri_list,
            "has_pet": has_pet_list,
            "has_both": has_both_list,
            "safetensor_file": safetensor_file_list,
        }, index=df.index)
        df = pd.concat([df, extra_df], axis=1)

        df.to_csv(self.output_csv, index=False)
        print(f"\n[Multimodal Tensor Processor Summary]")
        print(f"  • SafeTensors generated:       {count_created}")
        print(f"  • SafeTensors skipped:         {count_skipped}")
        print(f"  • Rows with both MRI & PET:    {sum(has_both_list)}")
        print(f"  • Rows with MRI only:          {sum(1 for m, p in zip(has_mri_list, has_pet_list) if m and not p)}")
        print(f"  • Rows with PET only:          {sum(1 for m, p in zip(has_mri_list, has_pet_list) if p and not m)}")
        print(f"  • Rows without imaging:        {count_no_files}")
        print(f"  • SafeTensors saved to:        {self.output_dir}")
        print(f"  • Updated dataset CSV saved to:{self.output_csv}")
