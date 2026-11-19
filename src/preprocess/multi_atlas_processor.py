from pathlib import Path
import json
import os
import shutil
import tarfile
from typing import Dict, List, Optional, Tuple
import urllib.request

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from src.utils import ParameterKeys


class MultiAtlasProcessor:
    """
    Automated Multi-Atlas Downloader & Standardizer (8 Atlases, 988 ROIs).
    Standardizes atlases to a target 1mm MNI152 reference grid (197x233x189).

    Reference:
        Song et al., "Developing a knowledge-guided federated graph attention learning network
        with a diffusion module to diagnose Alzheimer's disease",
        Medical Image Analysis, 107 (2026), 103794.
    """

    ATLAS_MANIFEST: Dict[str, dict] = {
        "AAL1": {
            "expected_rois": 116,
            "description": "Automated Anatomical Labeling v1",
            "url": "https://raw.githubusercontent.com/neurodata/neuroparc/master/atlases/label/Human/AAL_space-MNI152NLin6_res-1x1x1.nii.gz",
            "filename": "AAL1_raw.nii.gz",
            "format": "raw_nii",
        },
        "AAL2": {
            "expected_rois": 120,
            "description": "Automated Anatomical Labeling v2 (Rolls et al. 2015)",
            "url": "https://raw.githubusercontent.com/SCIInstitute/BrainStimulator/master/Data/Anatomical_Models/Colin27/Atlases/AAL2.nii.gz",
            "filename": "AAL2_raw.nii.gz",
            "format": "raw_nii",
        },
        "AAL3": {
            "expected_rois": 166,
            "description": "Automated Anatomical Labeling v3",
            "url": None,
            "filename": "AAL3v1_1mm_197x233x189.nii.gz",
            "format": "existing",
        },
        "Brodmann": {
            "expected_rois": 41,
            "description": "Brodmann Cytoarchitectonic Areas (Lacadie / NeuroParc)",
            "url": "https://raw.githubusercontent.com/neurodata/neuroparc/master/atlases/label/Human/Brodmann_space-MNI152NLin6_res-1x1x1.nii.gz",
            "filename": "Brodmann_raw.nii.gz",
            "format": "raw_nii",
        },
        "Hammersmith": {
            "expected_rois": 83,
            "description": "Hammersmith adult brain maximum probability map (n30r83)",
            "url": "https://raw.githubusercontent.com/neurodata/neuroparc/master/atlases/label/Human/Hammersmith_space-MNI152NLin6_res-1x1x1.nii.gz",
            "filename": "Hammersmith_raw.nii.gz",
            "format": "raw_nii",
        },
        "HarvardOxford": {
            "expected_rois": 113,
            "description": "Harvard-Oxford Cortical Lateralized (96) + Subcortical (17) MaxProb Thr25",
            "url": "https://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz",
            "filename": "HarvardOxford.tgz",
            "format": "harvard_oxford_archive",
        },
        "Juelich": {
            "expected_rois": 103,
            "description": "Juelich Histological Cytoarchitectonic Atlas (103 ROIs)",
            "url": "https://raw.githubusercontent.com/neurodata/neuroparc/master/atlases/label/Human/Juelich_space-MNI152NLin6_res-1x1x1.nii.gz",
            "filename": "Juelich_raw.nii.gz",
            "format": "raw_nii",
        },
        "Brainnetome": {
            "expected_rois": 246,
            "description": "Brainnetome Atlas (Fan et al. 2016)",
            "url": "https://raw.githubusercontent.com/fieldtrip/fieldtrip/master/template/atlas/brainnetome/BNA_MPM_thr25_1.25mm.nii",
            "filename": "Brainnetome_raw.nii",
            "format": "raw_nii",
        },
    }

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        # 1. Target reference template
        ref_path = self.parameters.get(
            ParameterKeys.TEMPLATE,
            self.parameters.get("reference", "data/processed/AAL3v1_1mm_197x233x189.nii.gz"),
        )
        self.reference_path = Path(ref_path)

        # 2. Output and raw download directories
        out_dir = self.parameters.get(
            ParameterKeys.OUT_DIR, "data/processed/atlases_1mm_197x233x189"
        )
        self.output_dir = Path(out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = self.parameters.get(
            "raw_dir", self.parameters.get(ParameterKeys.IN_DIR, "data/raw/atlases")
        )
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # 3. Execution flags
        self.skip_existing = self.parameters.get(ParameterKeys.SKIP, True)
        fsl_dir = self.parameters.get("fsl_dir", os.environ.get("FSLDIR", "/usr/local/fsl"))
        self.fsl_dir = Path(fsl_dir) if fsl_dir else None

    def _ensure_reference_exists(self):
        """
        Ensures target reference template grid (197, 233, 189) exists.
        If missing, automatically derives it from standard MNI152 1mm (nilearn)
        and resamples configs/AAL3/AAL3v1_1mm.nii.gz onto it.
        """
        if self.reference_path.exists():
            return

        print(f"[MultiAtlasProcessor] Reference grid not found at: {self.reference_path}")
        print("[MultiAtlasProcessor] Initializing standard MNI152 1mm reference template (197, 233, 189)...")

        import nilearn.datasets
        mni_img = nilearn.datasets.load_mni152_template(resolution=1)

        # Save MNI152 template if missing
        mni_standard_path = Path("data/processed/MNI152_T1_1mm.nii.gz")
        if not mni_standard_path.exists():
            mni_standard_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(mni_img, str(mni_standard_path))
            print(f"  ✓ Saved standard MNI152 template: {mni_standard_path}")

        # If reference is AAL3 resampled
        if "AAL3" in self.reference_path.name:
            aal3_raw = Path("configs/AAL3/AAL3v1_1mm.nii.gz")
            if not aal3_raw.exists():
                raise FileNotFoundError(f"Source AAL3 atlas not found at: {aal3_raw}")

            print(f"  * Resampling {aal3_raw.name} onto MNI152 grid (197, 233, 189)...")
            self._resample_atlas_to_reference(
                src_atlas_path=aal3_raw,
                reference_path=mni_standard_path,
                output_path=self.reference_path,
            )
            print(f"  ✓ Created reference atlas: {self.reference_path}")
        else:
            # Directly use MNI152 as reference
            self.reference_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(mni_img, str(self.reference_path))
            print(f"  ✓ Created reference grid: {self.reference_path}")

    def _download_file(self, url: str, dest_path: Path, timeout: int = 60) -> bool:
        """Downloads a remote file with User-Agent header and progress reporting."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and dest_path.stat().st_size > 1000:
            print(f"  [Cached] {dest_path.name} already exists ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")
            return True

        print(f"  [Downloading] {url} -> {dest_path.name}...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"  ✓ Download completed: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    def _extract_harvard_oxford(self, archive_path: Path, extract_dir: Path) -> Path:
        """
        Extracts Harvard-Oxford archive and merges cortical lateralized (96) + subcortical (17)
        into a unified 113-ROI integer-labeled volume.
        """
        extract_dir.mkdir(parents=True, exist_ok=True)
        merged_path = extract_dir / "HarvardOxford_merged.nii.gz"
        if merged_path.exists():
            return merged_path

        print("  [Extracting] Harvard-Oxford cortical lateralized and subcortical maps...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        # Use lateralized cortical maxprob-thr25 (96 regions)
        cortl_candidates = list(extract_dir.glob("**/*cortl*maxprob*thr25*1mm.nii*"))
        if not cortl_candidates:
            cortl_candidates = list(extract_dir.glob("**/*cort*maxprob*thr25*1mm.nii*"))

        sub_candidates = list(extract_dir.glob("**/*sub*maxprob*thr25*1mm.nii*"))

        if not cortl_candidates or not sub_candidates:
            raise FileNotFoundError(f"Could not locate Harvard-Oxford NIfTI files inside {extract_dir}")

        cort_img = nib.load(str(cortl_candidates[0]))
        sub_img = nib.load(str(sub_candidates[0]))

        cort_data = np.asanyarray(cort_img.dataobj).astype(np.int32)
        sub_data = np.asanyarray(sub_img.dataobj).astype(np.int32)

        # Exclude white matter (1, 12) and cerebral cortex (2, 13) from subcortical map
        # to preserve true subcortical structures without overwriting cortex
        sub_filtered = sub_data.copy()
        sub_filtered[(sub_data == 1) | (sub_data == 2) | (sub_data == 12) | (sub_data == 13)] = 0

        max_cort = int(np.max(cort_data))
        sub_vals = sorted([x for x in np.unique(sub_filtered) if x > 0])

        sub_remapped = np.zeros_like(sub_filtered)
        for i, v in enumerate(sub_vals, start=1):
            sub_remapped[sub_filtered == v] = max_cort + i

        # Merge: cortical regions have priority, subcortical fills remaining
        merged_data = np.where(cort_data > 0, cort_data, sub_remapped)
        merged_img = nib.Nifti1Image(merged_data, cort_img.affine, cort_img.header)
        nib.save(merged_img, str(merged_path))
        print(f"  ✓ Harvard-Oxford merged saved: {merged_path.name} (96 cortical + {len(sub_vals)} subcortical = {len(np.unique(merged_data)) - 1} ROIs)")
        return merged_path

    def _resample_atlas_to_reference(
        self,
        src_atlas_path: Path,
        reference_path: Path,
        output_path: Path,
    ) -> Tuple[Tuple[int, ...], int]:
        """
        Resamples source atlas onto reference geometry using Nearest-Neighbor
        interpolation (order=0) to strictly preserve discrete integer labels.
        """
        src_img = nib.load(str(src_atlas_path))
        ref_img = nib.load(str(reference_path))

        src_data = np.asanyarray(src_img.dataobj)
        if src_data.ndim > 3:
            src_data = src_data[..., 0]

        ref_shape = ref_img.shape[:3]
        ref_affine = ref_img.affine
        src_affine = src_img.affine

        transform_matrix = np.linalg.inv(src_affine) @ ref_affine

        resampled_data = affine_transform(
            src_data,
            matrix=transform_matrix[:3, :3],
            offset=transform_matrix[:3, 3],
            output_shape=ref_shape,
            order=0,
            mode="constant",
            cval=0,
        ).astype(np.int32)

        out_img = nib.Nifti1Image(resampled_data, ref_affine)
        out_img.set_data_dtype(np.int32)

        _, s_code = ref_img.get_sform(coded=True)
        _, q_code = ref_img.get_qform(coded=True)
        out_img.set_sform(ref_affine, code=int(s_code) if s_code else 2)
        out_img.set_qform(ref_affine, code=int(q_code) if q_code else 2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, str(output_path))

        unique_labels = sorted([int(x) for x in np.unique(resampled_data) if x > 0])
        return ref_shape, len(unique_labels)

    def launch(self) -> Path:
        self._ensure_reference_exists()

        ref_img = nib.load(str(self.reference_path))
        print("=" * 70)
        print("MULTI-ATLAS STANDARDIZATION PIPELINE (8 Atlases, 988 ROIs)")
        print("=" * 70)
        print(f"Reference Grid:   {self.reference_path.name}")
        print(f"  • Shape:        {ref_img.shape[:3]}")
        print(f"  • Resolution:   {ref_img.header.get_zooms()[:3]} mm")
        print(f"Output Directory: {self.output_dir.resolve()}\n")

        summary_records = []

        for atlas_name, info in self.ATLAS_MANIFEST.items():
            print(f"\n[{atlas_name}] Processing ({info['expected_rois']} expected ROIs)...")
            target_file = self.output_dir / f"{atlas_name}_1mm_197x233x189.nii.gz"

            if self.skip_existing and target_file.exists():
                existing_img = nib.load(str(target_file))
                unique_rois = len([x for x in np.unique(existing_img.get_fdata()) if x > 0])
                print(f"  ✓ [Skipping] Already exists: {target_file.name} ({unique_rois} ROIs)")
                summary_records.append({
                    "atlas": atlas_name,
                    "file": target_file.name,
                    "shape": str(existing_img.shape[:3]),
                    "rois": unique_rois,
                    "expected": info["expected_rois"],
                    "status": "CACHED",
                })
                continue

            # Case 1: AAL3
            if atlas_name == "AAL3":
                if "AAL3" in self.reference_path.name:
                    shutil.copy2(str(self.reference_path), str(target_file))
                    unique_rois = len([x for x in np.unique(ref_img.get_fdata()) if x > 0])
                    shape = ref_img.shape[:3]
                else:
                    aal3_raw = Path("configs/AAL3/AAL3v1_1mm.nii.gz")
                    if aal3_raw.exists():
                        shape, unique_rois = self._resample_atlas_to_reference(
                            src_atlas_path=aal3_raw,
                            reference_path=self.reference_path,
                            output_path=target_file,
                        )
                    else:
                        raise FileNotFoundError(f"Could not find AAL3 source file at {aal3_raw}")

                print(f"  ✓ AAL3 reference linked: {target_file.name} ({unique_rois} ROIs)")
                summary_records.append({
                    "atlas": atlas_name,
                    "file": target_file.name,
                    "shape": str(shape),
                    "rois": unique_rois,
                    "expected": info["expected_rois"],
                    "status": "READY",
                })
                continue

            # Case 2: Check FSL installation if available
            raw_source_file = self.raw_dir / info["filename"]
            if self.fsl_dir and self.fsl_dir.exists():
                fsl_atlas_dir = self.fsl_dir / "data" / "atlases"
                if atlas_name == "HarvardOxford":
                    ho_cort = fsl_atlas_dir / "HarvardOxford" / "HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz"
                    if ho_cort.exists():
                        raw_source_file = ho_cort
                elif atlas_name == "Juelich":
                    juel = fsl_atlas_dir / "Juelich" / "Juelich-maxprob-thr25-1mm.nii.gz"
                    if juel.exists():
                        raw_source_file = juel

            # Case 3: Download raw atlas if needed
            if not raw_source_file.exists():
                if info["url"] is not None:
                    success = self._download_file(info["url"], raw_source_file)
                    if not success:
                        print(f"  ⚠ Skipping {atlas_name} due to download failure.")
                        continue
                else:
                    print(f"  ⚠ No download URL provided for {atlas_name}.")
                    continue

            # Special format extraction (Harvard-Oxford archive)
            source_nii = raw_source_file
            if info["format"] == "harvard_oxford_archive":
                source_nii = self._extract_harvard_oxford(
                    archive_path=raw_source_file,
                    extract_dir=self.raw_dir / "HarvardOxford_extracted",
                )

            # Resample onto target geometry
            print(f"  [Resampling] {source_nii.name} -> {target_file.name} (Nearest-Neighbor)...")
            shape, rois = self._resample_atlas_to_reference(
                src_atlas_path=source_nii,
                reference_path=self.reference_path,
                output_path=target_file,
            )
            print(f"  ✓ Resampled shape: {shape}, Active ROIs retained: {rois}")

            summary_records.append({
                "atlas": atlas_name,
                "file": target_file.name,
                "shape": str(shape),
                "rois": rois,
                "expected": info["expected_rois"],
                "status": "READY",
            })

        # Save standardization manifest
        manifest_path = self.output_dir / "multi_atlas_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(summary_records, f, indent=2)

        print("\n" + "=" * 70)
        print("STANDARDIZATION SUMMARY")
        print("=" * 70)
        total_rois = sum(r["rois"] for r in summary_records)
        for r in summary_records:
            print(
                f"  • {r['atlas']:<15} | Shape: {r['shape']:<18} | "
                f"ROIs: {r['rois']:<3} / {r['expected']:<3} | Status: {r['status']}"
            )
        print("-" * 70)
        print(f"Total Standardized Multi-Atlas ROIs: {total_rois} (Target: 988)")
        print(f"Manifest saved to: {manifest_path.resolve()}\n")

        return self.output_dir
