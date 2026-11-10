import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from src.utils import ParameterKeys


class AtlasResamplerProcessor:
    """
    Resamples a discrete NIfTI brain atlas (e.g., AAL3v1_1mm) onto the exact
    spatial grid, affine, and orientation of a target MNI152 reference scan
    using Nearest-Neighbor interpolation (order=0) to preserve integer labels.
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        template_key = getattr(ParameterKeys, "TEMPLATE", "template") if ParameterKeys else "template"
        skip_key = getattr(ParameterKeys, "SKIP", "skip") if ParameterKeys else "skip"

        self.atlas_path = Path(
            self.parameters.get("atlas_path", "configs/AAL3/AAL3v1_1mm.nii.gz")
        )
        self.reference_path = Path(
            self.parameters.get(
                "reference_path",
                self.parameters.get(template_key, "data/processed/MNI152_T1_1mm.nii.gz"),
            )
        )
        self.output_path = Path(
            self.parameters.get(
                "output_path", "data/processed/AAL3v1_1mm_197x233x189.nii.gz"
            )
        )
        self.skip_existing = bool(
            self.parameters.get("skip_existing", self.parameters.get(skip_key, True))
        )

    def launch(self) -> Path:
        if not self.atlas_path.exists():
            raise FileNotFoundError(f"Source atlas not found: {self.atlas_path}")
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference MNI image not found: {self.reference_path}")

        if self.skip_existing and self.output_path.exists():
            print(f"[Atlas Resampler] Skipping (already exists): {self.output_path}")
            return self.output_path

        atlas_img = nib.load(str(self.atlas_path))
        ref_img = nib.load(str(self.reference_path))

        # Handle 4D reference volumes safely by taking the first 3D volume
        if ref_img.ndim > 3:
            ref_img = nib.Nifti1Image(ref_img.get_fdata()[..., 0], ref_img.affine)

        print("[Atlas Resampler] Starting atlas resampling...")
        print(
            f"  * Source atlas:   {self.atlas_path.name} "
            f"| shape={atlas_img.shape}, zooms={atlas_img.header.get_zooms()}, "
            f"axes={nib.aff2axcodes(atlas_img.affine)}"
        )
        print(
            f"  * Target grid:    {self.reference_path.name} "
            f"| shape={ref_img.shape}, zooms={ref_img.header.get_zooms()[:3]}, "
            f"axes={nib.aff2axcodes(ref_img.affine)}"
        )

        # Resample with order=0 (Nearest Neighbor) so discrete ROI IDs are preserved
        resampled_img = resample_from_to(atlas_img, ref_img, order=0)
        atlas_data = np.rint(resampled_img.get_fdata()).astype(np.int16)

        # Build clean int16 NIfTI without inheriting float scaling slopes from MRI headers
        out_img = nib.Nifti1Image(atlas_data, ref_img.affine)
        out_img.set_data_dtype(np.int16)

        _, s_code = ref_img.get_sform(coded=True)
        _, q_code = ref_img.get_qform(coded=True)
        out_img.set_sform(ref_img.affine, code=int(s_code) if s_code else 2)
        out_img.set_qform(ref_img.affine, code=int(q_code) if q_code else 2)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, str(self.output_path))

        # Post-resampling verification checks
        orig_labels = set(np.unique(np.rint(atlas_img.get_fdata()).astype(np.int16)))
        new_labels = set(np.unique(atlas_data))
        missing_labels = sorted(orig_labels - new_labels)

        ref_data = ref_img.get_fdata()
        fg_mask = atlas_data > 0
        brain_overlap = float(np.mean(ref_data[fg_mask] > 0) * 100.0) if np.any(fg_mask) else 0.0

        print("\n[Atlas Resampler Summary]")
        print(f"  * Output saved to:     {self.output_path}")
        print(f"  * Output shape:        {atlas_data.shape}")
        print(f"  * Output axes:         {nib.aff2axcodes(out_img.affine)}")
        print(f"  * Labels preserved:    {len(new_labels)}/{len(orig_labels)} (including 0 background)")
        if missing_labels:
            print(f"  * [Warning] Missing labels after resampling: {missing_labels}")
        print(f"  * Reference overlap:   {brain_overlap:.1f}% of atlas voxels cover non-zero reference voxels")

        return self.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample AAL3 atlas to match MNI-registered MRI/PET scan shape."
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default="configs/AAL3/AAL3v1_1mm.nii.gz",
        help="Path to the source atlas NIfTI file.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to one registered *_mni.nii.gz scan (shape 197x233x189) or MNI template.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/AAL3v1_1mm_197x233x189.nii.gz",
        help="Path where the resampled atlas will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output atlas if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processor = AtlasResamplerProcessor(
        {
            "atlas_path": args.atlas,
            "reference_path": args.reference,
            "output_path": args.output,
            "skip_existing": not args.overwrite,
        }
    )
    processor.launch()