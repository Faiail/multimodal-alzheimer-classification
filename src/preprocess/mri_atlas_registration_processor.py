from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
from src.utils import ParameterKeys


class MRIAtlasRegistrationProcessor:
    """
    Registers sMRI (T1 MPRAGE) scans to the MNI152 standard atlas space
    using SimpleITK (12-DOF affine transformation).
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        self.nifti_dir = Path(self.parameters.get(ParameterKeys.IN_DIR, "./"))
        self.output_dir = Path(self.parameters.get(ParameterKeys.OUT_DIR, "./"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = self.parameters.get(ParameterKeys.SKIP, True)

        template_path = self.parameters.get(
            ParameterKeys.TEMPLATE, "data/processed/MNI152_T1_1mm.nii.gz"
        )
        self.template_path = Path(template_path)

    def _is_mri(self, filename: str) -> bool:
        return "MPRAGE" in filename.upper()

    def _run_flirt_12dof(
        self,
        in_mri: Path,
        ref_template: Path | sitk.Image,
        out_mri: Path,
        out_mat: Path,
    ):
        """
        Executes 12-DOF affine registration via SimpleITK:
        rotation, translation, scaling, and shearing.
        """
        if isinstance(ref_template, sitk.Image):
            fixed = ref_template
        else:
            fixed = sitk.ReadImage(str(ref_template), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(in_mri), sitk.sitkFloat32)

        initial_tx = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.10, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)

        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        reg.SetOptimizerScalesFromPhysicalShift()

        # Multi-resolution pyramid (3 levels)
        reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        reg.SetInitialTransform(initial_tx, inPlace=False)

        final_tx = reg.Execute(fixed, moving)
        resampled = sitk.Resample(
            moving,
            fixed,
            final_tx,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )
        sitk.WriteImage(resampled, str(out_mri))
        sitk.WriteTransform(final_tx, str(out_mat))

    def launch(self):
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"MNI template not found at: {self.template_path}. "
                "Please verify that FSL is installed or provide 'template_path' explicitly."
            )

        all_files = sorted(list(self.nifti_dir.glob("*.nii*")))
        mri_files = [f for f in all_files if self._is_mri(f.name)]

        print(
            f"[Atlas Registration] Found {len(mri_files)} sMRI scans to register to MNI..."
        )
        print(f"  • Reference template: {self.template_path.name}")
        print(f"  • Output directory:   {self.output_dir}")

        count_done = 0
        count_skipped = 0

        fixed_template = sitk.ReadImage(str(self.template_path), sitk.sitkFloat32)

        for mri_file in tqdm(mri_files, desc="Registering sMRI -> MNI152"):
            clean_name = mri_file.name.replace(".nii.gz", "").replace(".nii", "")
            out_mri = self.output_dir / f"{clean_name}_mni.nii.gz"
            out_mat = self.output_dir / f"{clean_name}_mri2atlas.mat"

            # Skip if already processed
            if self.skip_existing and out_mri.exists() and out_mat.exists():
                count_skipped += 1
                continue

            self._run_flirt_12dof(
                in_mri=mri_file,
                ref_template=fixed_template,
                out_mri=out_mri,
                out_mat=out_mat,
            )
            count_done += 1

        print("\n[Atlas Registration Summary]")
        print(f"  • Volumes successfully registered: {count_done}")
        print(f"  • Volumes skipped (already exist): {count_skipped}")
        print(f"  • Transformation matrices (.mat) saved to: {self.output_dir}")
