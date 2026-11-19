from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from safetensors.torch import save_file
import torch
from tqdm import tqdm

from src.utils import ParameterKeys


class MultiAtlasTensorProcessor:
    """
    Multi-Atlas Tensorial Preprocessing with SafeTensors (8 Atlases, 988 ROIs).
    Reference:
        Song et al., "Developing a knowledge-guided federated graph attention learning network
        with a diffusion module to diagnose Alzheimer's disease",
        Medical Image Analysis, 107 (2026), 103794.

    For each scan in the dataset:
      - Extracts 988-dimensional ROI volume/intensity vector from sMRI in MNI152 space.
      - Extracts 988-dimensional ROI SUVR vector from PET in MNI152 space (normalized on cerebellum).
      - Serializes vectors into a compact .safetensors file named <patient_id><scan_id>.safetensors:
          * 'smri':       Tensor(988,)
          * 'pet':        Tensor(988,) (if available)
          * 'multimodal': Tensor(988, 2) (if both available - ideal for GNN node features)
          * 'concat':     Tensor(1976,)  (if both available - ideal for LassoNet / MLP)
      - Augments/updates the dataset CSV with columns:
          'has_mri', 'has_pet', 'has_both', 'multi_atlas_safetensor'.
    """

    ATLAS_NAMES: List[Tuple[str, int]] = [
        ("AAL1", 116),
        ("AAL2", 120),
        ("AAL3", 166),
        ("Brodmann", 41),
        ("Hammersmith", 83),
        ("HarvardOxford", 113),
        ("Juelich", 103),
        ("Brainnetome", 246),
    ]

    # AAL3 Cerebellar reference labels for PET SUVR normalization (labels 95 to 112)
    CEREBELLUM_LABELS: List[int] = list(range(95, 113))

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        # 1. Paths configuration
        raw_dataset_path = self.parameters.get(
            ParameterKeys.RAW_DATASET,
            self.parameters.get(ParameterKeys.DATA_ROOT, "data/processed/multimodal_raw_dataset.csv"),
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

        atlas_dir = self.parameters.get(
            ParameterKeys.ATLAS_DIR,
            self.parameters.get("atlas_dir", "data/processed/atlases_1mm_197x233x189"),
        )
        self.atlas_dir = Path(atlas_dir)

        out_dir = self.parameters.get(
            ParameterKeys.OUT_DIR, "data/processed/multi_atlas_safetensors"
        )
        self.output_dir = Path(out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        out_csv = self.parameters.get(
            ParameterKeys.OUT_CSV,
            str(self.raw_dataset_path.parent / f"{self.raw_dataset_path.stem}_multiatlas.csv"),
        )
        self.output_csv = Path(out_csv)

        # 2. Processing parameters
        self.skip_existing = self.parameters.get(ParameterKeys.SKIP, True)
        self.only_paired = self.parameters.get(ParameterKeys.ONLY_PAIRED, False)
        self.extract_pet = self.parameters.get(
            ParameterKeys.EXTRACT_PET,
            self.parameters.get("extract_pet", True),
        )
        self.smri_metric = self.parameters.get(
            ParameterKeys.SMRI_METRIC,
            self.parameters.get("smri_metric", "volume"),
        )  # 'volume' (mm^3) or 'mean'

    def _extract_id_tokens(self, row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts patient_id (e.g. '012_S_4849') and scan_id (e.g. 'I361978') from a row.
        Consistent with MultimodalTensorProcessor ID parsing.
        """
        raw_val = None
        for col in ["subject_id", "subject", "image_id", "filename", "ID"]:
            if col in row and pd.notna(row[col]):
                raw_val = str(row[col])
                break

        patient_id = None
        scan_id = None

        if raw_val:
            m_ptid = re.search(r"([0-9]{3}_S_[0-9]{4})", raw_val)
            if m_ptid:
                patient_id = m_ptid.group(1)

            m_scan = re.search(r"_(I[0-9]+)(?:_|\.|$)", raw_val)
            if m_scan:
                scan_id = m_scan.group(1)

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

    def _index_imaging_files(
        self,
    ) -> Tuple[Dict[str, Path], Dict[str, List[Path]], Dict[str, Path], Dict[str, List[Path]]]:
        """Indexes available MNI152 sMRI and PET files by scan_id and patient_id."""
        mri_by_scan: Dict[str, Path] = {}
        mri_by_patient: Dict[str, List[Path]] = {}
        if self.mri_dir.exists():
            for f in sorted(list(self.mri_dir.glob("*_mni.nii*")) + list(self.mri_dir.glob("*.nii*"))):
                if f.name.endswith(".mat") or f.is_dir():
                    continue
                m_scan = re.search(r"_(I[0-9]+)", f.name)
                if m_scan:
                    mri_by_scan[m_scan.group(1)] = f
                m_ptid = re.search(r"([0-9]{3}_S_[0-9]{4})", f.name)
                if m_ptid:
                    ptid = m_ptid.group(1)
                    if f not in mri_by_patient.get(ptid, []):
                        mri_by_patient.setdefault(ptid, []).append(f)

        pet_by_scan: Dict[str, Path] = {}
        pet_by_patient: Dict[str, List[Path]] = {}
        if self.pet_dir.exists():
            for f in sorted(list(self.pet_dir.glob("*_mni.nii*")) + list(self.pet_dir.glob("*.nii*"))):
                if f.name.endswith(".mat") or f.is_dir():
                    continue
                m_scan = re.search(r"_(I[0-9]+)", f.name)
                if m_scan:
                    pet_by_scan[m_scan.group(1)] = f
                m_ptid = re.search(r"([0-9]{3}_S_[0-9]{4})", f.name)
                if m_ptid:
                    ptid = m_ptid.group(1)
                    if f not in pet_by_patient.get(ptid, []):
                        pet_by_patient.setdefault(ptid, []).append(f)

        return mri_by_scan, mri_by_patient, pet_by_scan, pet_by_patient

    def _load_and_precompute_atlas_masks(
        self,
    ) -> Tuple[List[Tuple[str, int, Tuple[np.ndarray, ...]]], Optional[Tuple[np.ndarray, ...]]]:
        """
        Pre-loads all 8 standardized atlases into memory once and precomputes voxel coordinates.
        Returns ordered list of 988 ROI coordinate tuples and cerebellar reference mask.
        """
        print("[MultiAtlasTensorProcessor] Pre-loading 8 standardized atlases into RAM...")
        rois_info = []
        cerebellum_indices = None
        total_rois = 0

        for atlas_name, expected_count in self.ATLAS_NAMES:
            atlas_candidates = sorted(
                list(self.atlas_dir.glob(f"{atlas_name}_*.nii*"))
                + list(self.atlas_dir.glob(f"*{atlas_name}*.nii*"))
            )
            if not atlas_candidates:
                raise FileNotFoundError(
                    f"Standardized atlas for '{atlas_name}' not found in: {self.atlas_dir}.\n"
                    "Please ensure MultiAtlasProcessor has been executed."
                )

            atlas_path = atlas_candidates[0]
            atlas_img = nib.load(str(atlas_path))
            atlas_data = np.asanyarray(atlas_img.dataobj).astype(np.int32)
            if atlas_data.ndim > 3:
                atlas_data = atlas_data[..., 0]

            unique_labels = sorted([int(x) for x in np.unique(atlas_data) if x > 0])
            print(f"  • {atlas_name:<15}: Loaded {len(unique_labels)} active ROIs (Expected: {expected_count})")

            for lbl in unique_labels:
                coords = np.where(atlas_data == lbl)
                col_name = f"{atlas_name}_ROI_{lbl}"
                rois_info.append((col_name, lbl, coords))

            total_rois += len(unique_labels)

            if atlas_name == "AAL3":
                cereb_mask = np.isin(atlas_data, self.CEREBELLUM_LABELS)
                cerebellum_indices = np.where(cereb_mask)

        print(f"  ✓ Precomputed {total_rois} ROI index masks across 8 atlases (Target: 988)\n")
        return rois_info, cerebellum_indices

    def launch(self) -> Path:
        if not self.raw_dataset_path.exists():
            raise FileNotFoundError(f"Raw dataset CSV not found at: {self.raw_dataset_path}")

        # Auto-standardize atlases if missing or incomplete
        if not self.atlas_dir.exists() or len(list(self.atlas_dir.glob("*.nii*"))) < 8:
            print("[MultiAtlasTensorProcessor] Standardized atlases missing or incomplete in:")
            print(f"  {self.atlas_dir}")
            print("[MultiAtlasTensorProcessor] Automatically launching MultiAtlasProcessor to generate them...")
            from src.preprocess.multi_atlas_processor import MultiAtlasProcessor

            MultiAtlasProcessor(self.parameters).launch()

        print("=" * 70)
        print("MULTI-ATLAS SAFETENSOR PROCESSOR (8 Atlases, 988 ROIs)")
        print("=" * 70)
        print(f"  • Raw dataset:    {self.raw_dataset_path}")
        print(f"  • sMRI MNI dir:   {self.mri_dir}")
        print(f"  • PET MNI dir:    {self.pet_dir}")
        print(f"  • Atlas dir:      {self.atlas_dir}")
        print(f"  • SafeTensor dir: {self.output_dir}")
        print(f"  • Output CSV:     {self.output_csv}")
        print(f"  • sMRI Metric:    {str(self.smri_metric).upper()}")
        print(f"  • Extract PET:    {self.extract_pet}")
        print(f"  • Only Paired:    {self.only_paired}\n")

        # Index scans
        mri_by_scan, mri_by_patient, pet_by_scan, pet_by_patient = self._index_imaging_files()
        print(f"  • Found {len(mri_by_scan)} sMRI scans ({len(mri_by_patient)} unique patients)")
        print(f"  • Found {len(pet_by_scan)} PET scans ({len(pet_by_patient)} unique patients)\n")

        # Pre-load 8 atlases into memory once
        rois_info, cerebellum_indices = self._load_and_precompute_atlas_masks()
        num_rois = len(rois_info)

        df = pd.read_csv(self.raw_dataset_path, low_memory=False)
        print(f"  • Processing {len(df)} rows from dataset...")

        has_mri_list = []
        has_pet_list = []
        has_both_list = []
        safetensor_file_list = []

        count_created = 0
        count_skipped = 0
        count_no_files = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Multi-Atlas SafeTensors"):
            ptid, scan_id = self._extract_id_tokens(row)

            # Match sMRI file
            mri_path: Optional[Path] = None
            if scan_id and scan_id in mri_by_scan:
                mri_path = mri_by_scan[scan_id]
            elif ptid and ptid in mri_by_patient:
                mri_path = mri_by_patient[ptid][0]

            # Match PET file
            pet_path: Optional[Path] = None
            if self.extract_pet:
                if scan_id and scan_id in pet_by_scan:
                    pet_path = pet_by_scan[scan_id]
                elif ptid and ptid in pet_by_patient:
                    pet_path = pet_by_patient[ptid][0]

            has_mri = mri_path is not None and mri_path.exists()
            has_pet = pet_path is not None and pet_path.exists()
            has_both = has_mri and has_pet

            has_mri_list.append(has_mri)
            has_pet_list.append(has_pet)
            has_both_list.append(has_both)

            if (self.only_paired and not has_both) or (not has_mri and not has_pet):
                safetensor_file_list.append("")
                count_no_files += 1
                continue

            fname_ptid = ptid if ptid else "UNKNOWN"
            fname_scan = scan_id if scan_id else "I0"
            st_filename = f"{fname_ptid}{fname_scan}.safetensors"
            st_path = self.output_dir / st_filename
            safetensor_file_list.append(st_filename)

            if self.skip_existing and st_path.exists():
                count_skipped += 1
                continue

            tensors_to_save: Dict[str, torch.Tensor] = {}

            # 1. Extract sMRI vector (988,)
            smri_vec = None
            if has_mri and mri_path is not None:
                mri_img = nib.load(str(mri_path))
                mri_data = np.asanyarray(mri_img.dataobj).astype(np.float32)
                if mri_data.ndim > 3:
                    mri_data = mri_data[..., 0]

                # Voxel volume in mm^3 (1.0 mm^3 for 1mm MNI template)
                voxel_vol = float(np.prod(mri_img.header.get_zooms()[:3]))

                vals = np.zeros(num_rois, dtype=np.float32)
                for i, (_, _, coords) in enumerate(rois_info):
                    vox_vals = mri_data[coords]
                    if len(vox_vals) > 0:
                        if self.smri_metric == "volume":
                            vals[i] = float(np.sum(vox_vals)) * voxel_vol
                        else:
                            vals[i] = float(np.mean(vox_vals))
                    else:
                        vals[i] = 0.0

                smri_vec = torch.from_numpy(vals)
                tensors_to_save["smri"] = smri_vec

            # 2. Extract PET vector (988,)
            pet_vec = None
            if has_pet and pet_path is not None:
                pet_img = nib.load(str(pet_path))
                pet_data = np.asanyarray(pet_img.dataobj).astype(np.float32)
                if pet_data.ndim > 3:
                    pet_data = np.mean(pet_data, axis=-1).astype(np.float32)

                ref_mean = 1.0
                if cerebellum_indices is not None and len(cerebellum_indices[0]) > 0:
                    ref_mean = float(np.mean(pet_data[cerebellum_indices]))
                    ref_mean = max(ref_mean, 1e-6)

                vals_pet = np.zeros(num_rois, dtype=np.float32)
                for i, (_, _, coords) in enumerate(rois_info):
                    vox_vals = pet_data[coords]
                    vals_pet[i] = (float(np.mean(vox_vals)) / ref_mean) if len(vox_vals) > 0 else 0.0

                pet_vec = torch.from_numpy(vals_pet)
                tensors_to_save["pet"] = pet_vec

            # 3. Multimodal combined representations
            if has_both and smri_vec is not None and pet_vec is not None:
                # Shape (988, 2): column 0 = sMRI, column 1 = PET (ideal for GNN ROI nodes)
                tensors_to_save["multimodal"] = torch.stack([smri_vec, pet_vec], dim=1)
                # Shape (1976,): concatenated vector (ideal for LassoNet / MLP feature selection)
                tensors_to_save["concat"] = torch.cat([smri_vec, pet_vec], dim=0)

            save_file(tensors_to_save, str(st_path))
            count_created += 1

        # Augment dataframe with tracking columns
        df = df.copy()
        extra_df = pd.DataFrame(
            {
                "has_mri": has_mri_list,
                "has_pet": has_pet_list,
                "has_both": has_both_list,
                "multi_atlas_safetensor": safetensor_file_list,
            },
            index=df.index,
        )
        df = pd.concat([df, extra_df], axis=1)

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)

        print("\n" + "=" * 70)
        print("MULTI-ATLAS SAFETENSOR PROCESSOR SUMMARY")
        print("=" * 70)
        print(f"  • SafeTensors generated:       {count_created}")
        print(f"  • SafeTensors skipped:         {count_skipped}")
        print(f"  • Rows with both MRI & PET:    {sum(has_both_list)}")
        print(f"  • Rows with MRI only:          {sum(1 for m, p in zip(has_mri_list, has_pet_list) if m and not p)}")
        print(f"  • Rows with PET only:          {sum(1 for m, p in zip(has_mri_list, has_pet_list) if p and not m)}")
        print(f"  • Rows without imaging:        {count_no_files}")
        print(f"  • SafeTensors saved to:        {self.output_dir.resolve()}")
        print(f"  • Updated dataset CSV saved to:{self.output_csv.resolve()}\n")

        return self.output_dir
