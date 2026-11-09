from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from src.utils import ParameterKeys
import os


class CoregistrationProcessor:
    """
    Co-registra rigidamente (6-DOF) le PET sulle sMRI trovate nella STESSA cartella.
    Riconosce automaticamente le modalita' dal nome del file (MPRAGE vs PET/FDG).
    """

    def __init__(
        self,
        parameters: dict,
    ):
        self.parameters = parameters
        self.init()

    def init(self):
        print("Init genereal parameters...")
        self.nifti_dir = self.parameters.get(ParameterKeys.IN_DIR)
        self.out_dir = self.parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.skip_existing = self.parameters.get(ParameterKeys.SKIP, True)
        print("Done!")

    def _get_ptid(self, filename: str) -> str:
        """Estrae il PTID (es. '002_S_0295') dai primi 3 token separati da '_'."""
        parts = filename.split("_")
        return f"{parts[0]}_{parts[1]}_{parts[2]}"

    def _is_mri(self, filename: str) -> bool:
        return "MPRAGE" in filename.upper()

    def _is_pet(self, filename: str) -> bool:
        name_up = filename.upper()
        return "PET" in name_up or "FDG" in name_up

    def _ensure_3d(self, pet_path: Path) -> Path:
        """Se il PET e' 4D (multi-frame temporale), calcola la media e lo salva a 3D."""
        img = nib.load(str(pet_path))
        if img.ndim == 4 and img.shape[-1] > 1:
            data_3d = np.mean(img.get_fdata(), axis=-1).astype(np.float32)
            out_3d = f"{self.out_dir}/temp_{pet_path.name}"
            nib.save(nib.Nifti1Image(data_3d, img.affine, img.header), out_3d)
            return Path(out_3d)
        return pet_path

    def _run_flirt(self, in_pet: Path, ref_mri: Path, out_pet: Path):
        """Esegue co-registrazione rigida 6-DOF (Mutual Information, trilinear) tramite SimpleITK."""
        fixed = sitk.ReadImage(str(ref_mri), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(in_pet), sitk.sitkFloat32)

        initial_tx = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
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

        # Multi-resolution framework (piramide a 3 livelli)
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
        sitk.WriteImage(resampled, str(out_pet))

    def launch(self):
        print("Starting...")
        all_files = sorted(list(Path(self.nifti_dir).glob("*.nii*")))
        print(f"[Scanner] Found {len(all_files)} volumes in {self.nifti_dir}")

        # 1. Separa e indicizza le sMRI per PTID
        mri_dict = {}
        pet_files = []

        for f in all_files:
            if self._is_mri(f.name):
                ptid = self._get_ptid(f.name)
                if ptid not in mri_dict:
                    mri_dict[ptid] = f
            elif self._is_pet(f.name):
                pet_files.append(f)

        print(f"[Analysis] sMRI indexed: {len(mri_dict)} patients | PET to process: {len(pet_files)}")

        count_done = 0
        count_skipped = 0
        count_missing = 0

        # 2. Itera sulle PET e co-registra sull'sMRI del corrispondente PTID
        for pet_file in tqdm(pet_files, desc="Co-registration PET -> sMRI"):
            clean_name = pet_file.name.replace(".nii.gz", "").replace(".nii", "")
            out_file = f"{self.out_dir}/{clean_name}_coreg.nii.gz"

            if self.skip_existing and Path(out_file).exists():
                count_skipped += 1
                continue

            ptid = self._get_ptid(pet_file.name)
            if ptid not in mri_dict:
                count_missing += 1
                continue

            ref_mri = mri_dict[ptid]

            # Gestione 4D -> 3D
            pet_3d = self._ensure_3d(pet_file)

            # FLIRT
            self._run_flirt(in_pet=pet_3d, ref_mri=ref_mri, out_pet=out_file)
            count_done += 1

            # Pulizia file 3D temporaneo
            if pet_3d != pet_file and pet_3d.exists():
                pet_3d.unlink()

        print("\n[Summary]")
        print(f"  • Volumes completed:        {count_done}")
        print(f"  • Volumes already processed (skip): {count_skipped}")
        print(f"  • PET without sMRI found:      {count_missing}")
        print(f"  • Output folder:             {self.out_dir}")