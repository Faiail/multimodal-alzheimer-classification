from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
from src.utils import ParameterKeys


class PETAtlasRegistrationProcessor:
    """
    Warps co-registered PET scans to the MNI152 standard atlas space
    by applying the precomputed sMRI-to-Atlas affine transformation matrix (.mat)
    via SimpleITK.
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        # Input directory with co-registered PET scans
        pet_in_dir = self.parameters.get(
            "pet_coreg_dir", self.parameters.get(ParameterKeys.IN_DIR, "./")
        )
        self.pet_coreg_dir = Path(pet_in_dir)

        # Directory where MRIAtlasRegistrationProcessor saved *_mri2atlas.mat
        mri_mni_dir = self.parameters.get(ParameterKeys.MAPPING, "./")
        self.mri_mni_dir = Path(mri_mni_dir)

        # Output directory for warped PET scans
        self.output_dir = Path(self.parameters.get(ParameterKeys.OUT_DIR, "./"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.skip_existing = self.parameters.get(ParameterKeys.SKIP, True)

        template_path = self.parameters.get(
            ParameterKeys.TEMPLATE, "data/processed/MNI152_T1_1mm.nii.gz"
        )
        self.template_path = Path(template_path)

    def _get_ptid(self, filename: str) -> str:
        """Extracts PTID (e.g. '002_S_0295') from filename tokens."""
        parts = filename.split("_")
        return f"{parts[0]}_{parts[1]}_{parts[2]}"

    def _apply_transform(
        self,
        in_pet: Path,
        ref_template: Path | sitk.Image,
        in_mat: Path,
        out_pet: Path,
    ):
        """
        Applies the precomputed affine transformation (.mat) to the PET image
        resampling it directly onto the fixed MNI template grid.
        """
        if isinstance(ref_template, sitk.Image):
            fixed = ref_template
        else:
            fixed = sitk.ReadImage(str(ref_template), sitk.sitkFloat32)

        moving = sitk.ReadImage(str(in_pet), sitk.sitkFloat32)
        tx = sitk.ReadTransform(str(in_mat))

        resampled = sitk.Resample(
            moving,
            fixed,
            tx,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )
        sitk.WriteImage(resampled, str(out_pet))

    def launch(self):
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Reference template not found at: {self.template_path}."
            )

        all_pet_files = sorted(list(self.pet_coreg_dir.glob("*.nii*")))

        print(f"[PET Atlas Warp] Found {len(all_pet_files)} co-registered PET scans to warp...")
        print(f"  • Reference template: {self.template_path.name}")
        print(f"  • sMRI .mat directory:{self.mri_mni_dir}")
        print(f"  • Output directory:   {self.output_dir}")

        # Pre-load fixed reference template once
        fixed_template = sitk.ReadImage(str(self.template_path), sitk.sitkFloat32)

        count_done = 0
        count_skipped = 0
        count_missing = 0

        for pet_file in tqdm(all_pet_files, desc="Warping PET -> MNI152"):
            clean_name = pet_file.name.replace(".nii.gz", "").replace(".nii", "")
            out_pet = self.output_dir / f"{clean_name}_mni.nii.gz"

            # Skip if already processed
            if self.skip_existing and out_pet.exists():
                count_skipped += 1
                continue

            ptid = self._get_ptid(pet_file.name)

            # Match corresponding .mat file from MRIAtlasRegistrationProcessor
            mat_candidates = list(self.mri_mni_dir.glob(f"{ptid}*mri2atlas.mat"))
            if not mat_candidates:
                count_missing += 1
                continue

            in_mat = mat_candidates[0]

            self._apply_transform(
                in_pet=pet_file,
                ref_template=fixed_template,
                in_mat=in_mat,
                out_pet=out_pet,
            )
            count_done += 1

        print("\n[PET Atlas Warp Summary]")
        print(f"  • PET volumes successfully warped to MNI: {count_done}")
        print(f"  • Volumes skipped (already exist):       {count_skipped}")
        print(f"  • PET scans missing sMRI .mat matrix:     {count_missing}")
        print(f"  • Warped PET volumes saved to:            {self.output_dir}")