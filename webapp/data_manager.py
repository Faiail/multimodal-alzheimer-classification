"""
Data Manager for Multimodal Alzheimer's Disease Scans.
Handles dataset indexing, patient filtering, 3D tensor slicing (axial, coronal, sagittal),
colormap application, and multimodal PET/MRI fusion rendering.
"""

import io
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as cm

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


class DataManager:
    COLORMAPS = {
        "gray": None,
        "inferno": cm.inferno,
        "hot": cm.hot,
        "jet": cm.jet,
        "viridis": cm.viridis,
        "plasma": cm.plasma
    }

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.dataset_csv = self.project_root / "data/processed/multimodal_raw_dataset.csv"
        self.tensors_dir = self.project_root / "data/processed/multimodal-tensors"
        self.df = pd.DataFrame()
        self.scan_cache = {}
        self.load_dataset()

    def load_dataset(self):
        if self.dataset_csv.exists():
            self.df = pd.read_csv(self.dataset_csv, low_memory=False)
            # Normalize boolean columns
            for col in ["has_mri", "has_pet", "has_both"]:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(bool)
            print(f"[DataManager] Successfully loaded {len(self.df)} scans ({self.df['PTID'].nunique()} unique patients).")
        else:
            print(f"[DataManager] Warning: Dataset CSV not found at {self.dataset_csv}")

    def get_stats(self) -> Dict[str, Any]:
        if self.df.empty:
            return {}
        return {
            "total_scans": len(self.df),
            "unique_patients": int(self.df["PTID"].nunique()),
            "classes": self.df["original_class"].value_counts().to_dict(),
            "has_mri_count": int(self.df["has_mri"].sum()),
            "has_pet_count": int(self.df["has_pet"].sum()),
            "paired_both_count": int(self.df["has_both"].sum()),
            "age_mean": round(float(self.df["AGE"].mean()), 1),
            "mmse_mean": round(float(self.df["MMSE"].mean()), 1)
        }

    def get_curated_showcase(self) -> List[Dict[str, Any]]:
        """Returns 6 curated representative patient scans across diagnoses and modalities."""
        if self.df.empty:
            return []
        
        candidates = [
            # 1. Classic CN with paired PET and MRI (Normal cognition)
            self.df[self.df["has_both"] & (self.df["original_class"] == "CN") & (self.df["MMSE"] >= 29)],
            # 2. Classic AD with paired PET and MRI (Hypometabolism + severe atrophy)
            self.df[self.df["has_both"] & (self.df["original_class"] == "AD") & (self.df["MMSE"] <= 24) & (self.df["APOE4"] >= 1)],
            # 3. MCI with paired PET (Cognitive impairment transition)
            self.df[self.df["has_both"] & (self.df["original_class"] == "MCI") & (self.df["MMSE"].between(25, 27))],
            # 4. Young-onset / Early stage patient
            self.df[self.df["has_both"] & (self.df["AGE"] < 65)],
            # 5. sMRI-only Normal control
            self.df[(~self.df["has_both"]) & (self.df["original_class"] == "CN")],
            # 6. sMRI-only Severe AD case
            self.df[(~self.df["has_both"]) & (self.df["original_class"] == "AD")]
        ]
        
        showcase = []
        for subset in candidates:
            if not subset.empty:
                row = subset.iloc[0]
                showcase.append(self._format_scan_record(row))
        return showcase

    def filter_scans(
        self,
        query: str = "",
        diagnosis: str = "ALL",
        modality: str = "ALL",
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        if self.df.empty:
            return {"total": 0, "scans": []}

        sub = self.df

        if query:
            q = query.strip().lower()
            mask = (
                sub["PTID"].astype(str).str.lower().str.contains(q)
                | sub["scan_id"].astype(str).str.lower().str.contains(q)
                | sub["safetensor_file"].astype(str).str.lower().str.contains(q)
            )
            sub = sub[mask]

        if diagnosis != "ALL":
            sub = sub[sub["original_class"] == diagnosis.upper()]

        if modality == "PAIRED":
            sub = sub[sub["has_both"]]
        elif modality == "MRI_ONLY":
            sub = sub[~sub["has_both"]]

        total = len(sub)
        paged = sub.iloc[offset : offset + limit]

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "scans": [self._format_scan_record(r) for _, r in paged.iterrows()]
        }

    def get_scan_by_id(self, scan_id: str) -> Optional[Dict[str, Any]]:
        if self.df.empty:
            return None
        
        # Match by scan_id or safetensor_file or PTID
        m = self.df[self.df["scan_id"] == scan_id]
        if m.empty:
            m = self.df[self.df["safetensor_file"] == scan_id]
        if m.empty and "." not in scan_id:
            m = self.df[self.df["safetensor_file"] == f"{scan_id}.safetensors"]
        if m.empty:
            m = self.df[self.df["PTID"] == scan_id]

        if not m.empty:
            return self._format_scan_record(m.iloc[0])
        return None

    def _format_scan_record(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "ptid": str(row.get("PTID", "Unknown")),
            "scan_id": str(row.get("scan_id", "I0")),
            "safetensor_file": str(row.get("safetensor_file", "")),
            "diagnosis": str(row.get("original_class", "CN")),
            "label": int(row.get("label", 0)),
            "age": float(row.get("AGE", 0.0)),
            "gender": str(row.get("PTGENDER", "Unknown")),
            "education": float(row.get("PTEDUCAT", 0.0)),
            "apoe4": float(row.get("APOE4", 0.0)),
            "mmse": float(row.get("MMSE", 0.0)),
            "has_mri": bool(row.get("has_mri", False)),
            "has_pet": bool(row.get("has_pet", False)),
            "has_both": bool(row.get("has_both", False)),
            "mri_file": str(row.get("mri_file", "")),
            "pet_file": str(row.get("pet_file", ""))
        }

    def load_scan_tensors(self, safetensor_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if safetensor_file in self.scan_cache:
            return self.scan_cache[safetensor_file]

        path = self.tensors_dir / safetensor_file
        if not path.exists():
            return None, None

        if load_file is None:
            return None, None

        try:
            data = load_file(str(path))
            mri = data["mri"].squeeze(0).numpy() if "mri" in data else None
            pet = data["pet"].squeeze(0).numpy() if "pet" in data else None
            
            # Simple LRU cache for 15 scans
            if len(self.scan_cache) > 15:
                oldest = next(iter(self.scan_cache))
                del self.scan_cache[oldest]
                
            self.scan_cache[safetensor_file] = (mri, pet)
            return mri, pet
        except Exception as e:
            print(f"[DataManager] Error loading {path}: {e}")
            return None, None

    def render_slice(
        self,
        safetensor_file: str,
        modality: str = "mri",
        plane: str = "axial",
        slice_idx: int = 64,
        colormap: str = "inferno",
        alpha: float = 0.5,
        window: float = 1.0,
        level: float = 0.0
    ) -> bytes:
        """
        Renders a 2D slice from 3D volume as PNG bytes.
        Supports:
          - planes: 'axial' (Z), 'coronal' (Y), 'sagittal' (X)
          - modalities: 'mri', 'pet', 'fused'
          - colormaps: 'gray', 'inferno', 'hot', 'jet', 'viridis', 'plasma'
          - fusion with alpha blending
        """
        mri, pet = self.load_scan_tensors(safetensor_file)
        slice_idx = max(0, min(127, int(slice_idx)))

        def extract_plane(vol: np.ndarray, p: str, idx: int) -> np.ndarray:
            if p == "sagittal":
                return vol[idx, :, :].T  # Orient for medical convention
            elif p == "coronal":
                return vol[:, idx, :].T
            else:  # axial
                return vol[:, :, idx].T

        if modality == "mri" or pet is None:
            if mri is None:
                return self._create_placeholder_png("sMRI Scan Not Available")
            sl = extract_plane(mri, plane, slice_idx)
            # Normalize to 0-1 with window/level
            sl_norm = np.clip((sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * window + level, 0.0, 1.0)
            uint8_img = (sl_norm * 255).astype(np.uint8)
            img = Image.fromarray(uint8_img)

        elif modality == "pet":
            sl = extract_plane(pet, plane, slice_idx)
            sl_norm = np.clip((sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * window + level, 0.0, 1.0)
            cmap_func = self.COLORMAPS.get(colormap, cm.inferno)
            if cmap_func is None:
                uint8_img = (sl_norm * 255).astype(np.uint8)
                img = Image.fromarray(uint8_img)
            else:
                colored = cmap_func(sl_norm)[:, :, :3]
                uint8_img = (colored * 255).astype(np.uint8)
                img = Image.fromarray(uint8_img)

        elif modality in ("fused", "fusion"):
            # MRI anatomical background + PET metabolic overlay
            if mri is None:
                return self._create_placeholder_png("MRI/PET Volume Missing")
            mri_sl = extract_plane(mri, plane, slice_idx)
            pet_sl = extract_plane(pet, plane, slice_idx)

            mri_norm = np.clip((mri_sl - mri_sl.min()) / (mri_sl.max() - mri_sl.min() + 1e-8), 0.0, 1.0)
            pet_norm = np.clip((pet_sl - pet_sl.min()) / (pet_sl.max() - pet_sl.min() + 1e-8), 0.0, 1.0)

            mri_rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
            cmap_func = self.COLORMAPS.get(colormap, cm.inferno) or cm.inferno
            pet_rgb = cmap_func(pet_norm)[:, :, :3]

            # Mask out background where PET uptake is near zero for crisp anatomical visibility
            pet_mask = (pet_norm > 0.08)[:, :, None].astype(float)
            effective_alpha = alpha * pet_mask

            fused = (1.0 - effective_alpha) * mri_rgb + effective_alpha * pet_rgb
            uint8_img = (np.clip(fused, 0.0, 1.0) * 255).astype(np.uint8)
            img = Image.fromarray(uint8_img)
        else:
            return self._create_placeholder_png("Invalid Modality")

        # Resize from 128x128 to 256x256 for smooth crisp display
        img = img.resize((256, 256), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _create_placeholder_png(self, message: str) -> bytes:
        img = Image.new("RGB", (256, 256), color=(20, 24, 33))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
