import nipype.interfaces.fsl as fsl
from pathlib import Path
import os
import traceback
import yaml
import nibabel as nib
import numpy as np

import logging
logging.getLogger('nipype').setLevel(logging.WARNING)
logging.getLogger('nipype.interface').setLevel(logging.WARNING)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_pet_4d(pet_file, output_dir):
    """
    Check if PET is 4D and extract/average to 3D if needed.
    
    Args:
        pet_file: Path to PET NIfTI file
        output_dir: Directory for preprocessed output
    
    Returns:
        Path to 3D PET file (either original if already 3D, or preprocessed)
    """
    img = nib.load(str(pet_file))
    
    if len(img.shape) > 3 and img.shape[3] > 1:
        print(f"    PET is 4D ({img.shape}), averaging across timepoints...")
        
        data_4d = img.get_fdata()
        data_3d = np.mean(data_4d, axis=3)
        
        img_3d = nib.Nifti1Image(data_3d, img.affine, img.header)
        
        output_3d = output_dir / f"{pet_file.stem}_3d.nii.gz"
        nib.save(img_3d, str(output_3d))
        
        print(f"    Saved 3D averaged PET: {output_3d.name}")
        return output_3d
    else:
        print(f"    PET is already 3D ({img.shape})")
        return pet_file

def process_single_pet(pet_file, reference_mri_file, subject_id, pet_scan_id, config, skip_existing=True):
    """
    Co-register a single PET image to its corresponding MRI.
    
    Args:
        pet_file: Path to input PET NIfTI file
        reference_mri_file: Path to reference MRI (N4-corrected)
        subject_id: MRI subject identifier (for organization)
        pet_scan_id: Unique PET scan identifier (includes date/series)
        config: Configuration dictionary
        skip_existing: Skip if output already exists
    
    Returns:
        dict with status and output paths
    """
    
    pet_coreg_dir = Path(config['paths']['processed']) / 'pet_coregistered'
    os.makedirs(pet_coreg_dir, exist_ok=True)
    
    pet_temp_dir = pet_coreg_dir / 'temp'
    os.makedirs(pet_temp_dir, exist_ok=True)
    
    output_pet_coreg = pet_coreg_dir / f'{pet_scan_id}_coreg.nii.gz'
    
    result = {
        'subject_id': subject_id,
        'pet_scan_id': pet_scan_id,
        'status': 'pending',
        'output_file': None,
        'error': None
    }
    
    try:
        if skip_existing and output_pet_coreg.exists():
            file_size = output_pet_coreg.stat().st_size
            if file_size > 0:
                print(f"  Co-registered PET exists ({file_size / 1024 / 1024:.1f} MB), skipping")
                result['output_file'] = output_pet_coreg
                result['status'] = 'success'
                return result
            else:
                print(f"  Co-registered PET exists but is empty, re-processing...")
        
        if not pet_file.exists():
            raise FileNotFoundError(f"PET file not found: {pet_file}")
        if not reference_mri_file.exists():
            raise FileNotFoundError(f"Reference MRI not found: {reference_mri_file}")
        
        print(f"  Running FSL FLIRT co-registration...")
        print(f"    PET: {pet_file.name}")
        print(f"    MRI: {reference_mri_file.name}")
        
        pet_3d_file = preprocess_pet_4d(pet_file, pet_temp_dir)
        
        print(f"    Output: {output_pet_coreg.name}")
        
        output_matrix = pet_coreg_dir / f'{pet_scan_id}_coreg.mat'
        
        coreg = fsl.FLIRT()
        coreg.inputs.in_file = str(pet_3d_file)
        coreg.inputs.reference = str(reference_mri_file)
        coreg.inputs.out_file = str(output_pet_coreg)
        coreg.inputs.out_matrix_file = str(output_matrix)
        coreg.inputs.dof = 6
        coreg.inputs.cost = 'mutualinfo'
        coreg.inputs.interp = 'trilinear'
        
        coreg.run()
        
        if pet_3d_file != pet_file and pet_3d_file.exists():
            os.remove(pet_3d_file)
        
        if output_pet_coreg.exists():
            file_size = output_pet_coreg.stat().st_size
            if file_size > 0:
                result['output_file'] = output_pet_coreg
                result['status'] = 'success'
                print(f"  ✓ Co-registration completed ({file_size / 1024 / 1024:.1f} MB)")
            else:
                raise FileNotFoundError(f"FLIRT created empty file: {output_pet_coreg}")
        else:
            raise FileNotFoundError(f"FLIRT completed but output not found: {output_pet_coreg}")
        
        return result
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"  ✗ ERROR: {e}")
        traceback.print_exc()
        return result

if __name__ == '__main__':
    print("--- Starting Single Subject PET Co-registration ---")
    
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'processing_config.yaml'
    config = load_config(config_path)
    
    base_dir = Path(config['paths']['processed'])
    pet_file = base_dir / 'nifti/002_S_1155_ADNI_Brain_PET__Raw_FDG_20110107_I209934.nii.gz'
    reference_mri = base_dir / 'n4/002_S_1155_MPRAGE_20111222_I274154_n4.nii.gz'
    subject_id = '002_S_1155_MPRAGE_20111222_I274154'  # use mri subject id
    pet_scan_id = '002_S_1155_ADNI_Brain_PET__Raw_FDG_20110107_I209934'  # unique pet scan id
    
    result = process_single_pet(pet_file, reference_mri, subject_id, pet_scan_id, config)
    
    if result['status'] == 'success':
        print(f"\n--- Processing Finished Successfully! ---")
        print(f"Output: {result['output_file']}")
    else:
        print(f"\n--- Processing Failed: {result['error']} ---")