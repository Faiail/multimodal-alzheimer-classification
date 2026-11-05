from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
from tqdm import tqdm
from src.utils import ParameterKeys


class CoregistrationProcessor:
    """
    Co-registra rigidamente (6-DOF) le PET sulle sMRI trovate nella STESSA cartella.
    Riconosce automaticamente le modalita' dal nome del file (MPRAGE vs PET/FDG).
    """

    def __init__(
        self,
        parameters: dict,
        nifti_dir: str | Path,
        output_dir: str | Path,
        skip_existing: bool = True,
    ):
        self.parameters = parameters
        self.init()

    def init(self):
        self.nifti_dir = self.parameters.get(ParameterKeys.IN_DIR)
        self.out_dir = self.parameters.get(ParameterKeys.OUT_DIR, "./")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = self.parameters.get(ParameterKeys.SKIP, True)

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
            out_3d = self.out_dir / f"temp_{pet_path.name}"
            nib.save(nib.Nifti1Image(data_3d, img.affine, img.header), str(out_3d))
            return out_3d
        return pet_path

    def _run_flirt(self, in_pet: Path, ref_mri: Path, out_pet: Path):
        """Esegue FSL FLIRT (6-DOF, mutual information)."""
        cmd = [
            "flirt",
            "-in", str(in_pet),
            "-ref", str(ref_mri),
            "-out", str(out_pet),
            "-dof", "6",
            "-cost", "mutualinfo",
            "-interp", "trilinear",
        ]
        subprocess.run(cmd, check=True)

    def launch(self):
        all_files = sorted(list(self.nifti_dir.glob("*.nii*")))
        print(f"[Scanner] Trovati {len(all_files)} volumi in {self.nifti_dir}")

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

        print(f"[Analisi] sMRI indicizzate: {len(mri_dict)} soggetti | PET da processare: {len(pet_files)}")

        count_done = 0
        count_skipped = 0
        count_missing = 0

        # 2. Itera sulle PET e co-registra sull'sMRI del corrispondente PTID
        for pet_file in tqdm(pet_files, desc="Co-registrazione PET -> sMRI"):
            clean_name = pet_file.name.replace(".nii.gz", "").replace(".nii", "")
            out_file = self.out_dir / f"{clean_name}_coreg.nii.gz"

            if self.skip_existing and out_file.exists():
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

        print("\n[Riepilogo Co-registrazione]")
        print(f"  • Scansioni completate:        {count_done}")
        print(f"  • Scansioni gia' fatte (skip): {count_skipped}")
        print(f"  • PET senza sMRI trovata:      {count_missing}")
        print(f"  • Cartella output:             {self.out_dir}")