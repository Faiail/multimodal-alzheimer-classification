"""
Script to visualize the quality of medical image preprocessing.
Generates a figure with 3 panels:
- A. MRI corrected with N4
- B. Segmentation with AAL3 atlas
- C. PET co-registration on MRI
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from nilearn import plotting

from src.utils.general import DEFAULT_DATA_ROOT, resolve_data_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(
    base_dir: Path = Path(DEFAULT_DATA_ROOT),
    config_dir: Path = PROJECT_ROOT / "configs",
    output_dir: Path = PROJECT_ROOT / "reports" / "figures",
):
    base_dir = Path(resolve_data_path(base_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n4_files = sorted(glob.glob(str(base_dir / "n4" / "*_n4.nii.gz")))
    if not n4_files:
        raise FileNotFoundError(f"No MRI N4 file found in {base_dir / 'n4'}")

    mri_path = Path(n4_files[0])
    subj_id = mri_path.name.replace("_n4.nii.gz", "")
    print(f"📊 Using subject: {subj_id}")

    atlas_path = config_dir / "AAL3" / "ROI_MNI_V7_1mm.nii"
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    pet_files = glob.glob(
        str(base_dir / "pet_coregistered" / f"*{subj_id.split('_MPRAGE')[0]}*_coreg.nii.gz")
    )
    pet_path = Path(pet_files[0]) if pet_files else None
    if pet_path is None:
        print("⚠️  No PET found for this subject, panel C will be skipped")

    print("🖼️  Generating visualizations...")
    output_files = []

    print("  - Image A: MRI N4")
    fig_mri = plt.figure(figsize=(10, 8))
    display_mri = plotting.plot_anat(
        str(mri_path),
        title="MRI (N4 Corrected)",
        cut_coords=None,
        draw_cross=False,
        annotate=True,
        cmap="gray",
        figure=fig_mri,
    )
    output_mri = output_dir / f"{subj_id}_01_mri_n4.png"
    plt.savefig(output_mri, dpi=300, bbox_inches="tight")
    print(f"    ✅ Saved: {output_mri.name}")
    output_files.append(output_mri)
    plt.close(fig_mri)

    print("  - Image B: AAL3 Atlas")
    fig_atlas = plt.figure(figsize=(10, 8))
    plotting.plot_roi(
        str(atlas_path),
        bg_img=str(mri_path),
        title="Atlas ROI Segmentation (AAL3)",
        cut_coords=display_mri.cut_coords,
        draw_cross=False,
        cmap="Paired",
        alpha=0.5,
        annotate=True,
        figure=fig_atlas,
    )
    output_atlas = output_dir / f"{subj_id}_02_atlas_segmentation.png"
    plt.savefig(output_atlas, dpi=300, bbox_inches="tight")
    print(f"    ✅ Saved: {output_atlas.name}")
    output_files.append(output_atlas)
    plt.close(fig_atlas)

    if pet_path and pet_path.exists():
        print("  - Image C: PET Co-registration")
        fig_pet = plt.figure(figsize=(10, 8))
        plotting.plot_stat_map(
            str(pet_path),
            bg_img=str(mri_path),
            title="PET Co-registration",
            cut_coords=display_mri.cut_coords,
            draw_cross=False,
            threshold="auto",
            cmap="hot",
            alpha=0.7,
            annotate=True,
            colorbar=True,
            figure=fig_pet,
        )
        output_pet = output_dir / f"{subj_id}_03_pet_coregistered.png"
        plt.savefig(output_pet, dpi=300, bbox_inches="tight")
        print(f"    ✅ Saved: {output_pet.name}")
        output_files.append(output_pet)
        plt.close(fig_pet)

    return output_files


if __name__ == "__main__":
    main()
