import nipype.interfaces.fsl as fsl
from pathlib import Path
import os
import nibabel as nib
import numpy as np
import pandas as pd
import traceback
import yaml

import logging
logging.getLogger('nipype').setLevel(logging.WARNING)
logging.getLogger('nipype.interface').setLevel(logging.WARNING)
logging.getLogger('nipype.workflow').setLevel(logging.WARNING)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def resample_atlas_to_image(atlas_file, reference_image, output_file):
    """Resample atlas to match the dimensions and resolution of reference image."""
    print(f"    Resampling atlas to match {reference_image.name}...")
    flirt = fsl.FLIRT(
        in_file=str(atlas_file),
        reference=str(reference_image),
        out_file=str(output_file),
        apply_xfm=True,
        uses_qform=True,
        interp='nearestneighbour'
    )
    flirt.run()
    return output_file

def extract_volumes(image_file, atlas_file, region_labels, temp_dir='.'):
    """Calculates the volume in mm^3 for each atlas region in a given image."""
    volumes = {}
    temp_dir = Path(temp_dir)
    
    for label in region_labels:
        mask_file = temp_dir / f'temp_mask_{label}.nii.gz'
        masker = fsl.ImageMaths(
            in_file=str(atlas_file),
            op_string=f'-thr {label} -uthr {label} -bin',
            out_file=str(mask_file)
        )
        masker.run()

        stats = fsl.ImageStats(
            in_file=str(image_file),
            op_string='-k %s -V',
            mask_file=str(mask_file)
        )
        results = stats.run()
        volume_mm3 = results.outputs.out_stat[1] if len(results.outputs.out_stat) > 1 else 0
        volumes[label] = volume_mm3
        
        if mask_file.exists():
            os.remove(mask_file)
    
    return volumes

def extract_suvr(pet_file, atlas_file, region_labels, reference_labels, temp_dir='.'):
    """Calculates the SUVR for each atlas region in a PET image."""
    suvrs = {}
    temp_dir = Path(temp_dir)
    
    temp_masks = []
    for i, label in enumerate(reference_labels):
        mask_file = temp_dir / f'ref_mask_part_{i}.nii.gz'
        masker = fsl.ImageMaths(
            in_file=str(atlas_file),
            op_string=f'-thr {label} -uthr {label} -bin',
            out_file=str(mask_file)
        )
        masker.run()
        temp_masks.append(mask_file)
    
    reference_mask = temp_dir / 'reference_mask.nii.gz'
    if len(temp_masks) == 1:
        os.rename(temp_masks[0], reference_mask)
    else:
        combined = fsl.ImageMaths(
            in_file=str(temp_masks[0]),
            op_string='',
            out_file=str(reference_mask)
        )
        combined.run()
        
        for mask in temp_masks[1:]:
            adder = fsl.ImageMaths(
                in_file=str(reference_mask),
                op_string=f'-add {str(mask)}',
                out_file=str(reference_mask)
            )
            adder.run()
            if mask.exists():
                os.remove(mask)
        
        if temp_masks[0].exists():
            os.remove(temp_masks[0])
        
        binarizer = fsl.ImageMaths(
            in_file=str(reference_mask),
            op_string='-bin',
            out_file=str(reference_mask)
        )
        binarizer.run()

    ref_stats = fsl.ImageStats(
        in_file=str(pet_file),
        op_string=f'-k {str(reference_mask)} -M',
        mask_file=str(reference_mask)
    )
    ref_mean = ref_stats.run().outputs.out_stat
    
    if ref_mean == 0:
        print("    WARNING: Reference mean is zero. Cannot calculate SUVR.")
        if reference_mask.exists():
            os.remove(reference_mask)
        return None

    for label in region_labels:
        mask_file = temp_dir / f'temp_mask_{label}.nii.gz'
        masker = fsl.ImageMaths(
            in_file=str(atlas_file),
            op_string=f'-thr {label} -uthr {label} -bin',
            out_file=str(mask_file)
        )
        masker.run()

        roi_stats = fsl.ImageStats(
            in_file=str(pet_file),
            op_string=f'-k {str(mask_file)} -M',
            mask_file=str(mask_file)
        )
        roi_mean = roi_stats.run().outputs.out_stat
        
        suvr = roi_mean / ref_mean
        suvrs[label] = suvr
        
        if mask_file.exists():
            os.remove(mask_file)

    if reference_mask.exists():
        os.remove(reference_mask)
    return suvrs

def extract_features_single_subject(mri_subject_id, pet_scan_ids, config, skip_existing=True):
    """
    Extract features for a single MRI subject with zero or more PET scans.
    Returns features as DataFrame instead of saving to individual CSVs.
    """
    
    processed_dir = Path(config['paths']['processed'])
    
    temp_dir = Path(config['paths']['features']) / 'temp_atlas' / mri_subject_id
    os.makedirs(temp_dir, exist_ok=True)
    
    result = {
        'mri_subject_id': mri_subject_id,
        'pet_scan_ids': pet_scan_ids,
        'status': 'pending',
        'features_df': None,  # return dataframe instead of file paths
        'error': None
    }
    
    try:
        seg_dir = processed_dir / 'segmentation' / mri_subject_id
        
        gm_files = list(seg_dir.glob(f'wc1*{mri_subject_id}*.nii'))
        wm_files = list(seg_dir.glob(f'wc2*{mri_subject_id}*.nii'))
        csf_files = list(seg_dir.glob(f'wc3*{mri_subject_id}*.nii'))
        
        if not (gm_files and wm_files and csf_files):
            raise FileNotFoundError(
                f"Tissue maps not found for {mri_subject_id}. "
                f"Found: GM={len(gm_files)}, WM={len(wm_files)}, CSF={len(csf_files)}"
            )
        
        gm_file = gm_files[0]
        wm_file = wm_files[0]
        csf_file = csf_files[0]
        
        print(f"  Found tissue maps for {mri_subject_id}")
        
        atlas_file = Path(config['atlas']['aal_file'])
        resampled_atlas = temp_dir / 'atlas_resampled.nii.gz'
        
        if not resampled_atlas.exists():
            resample_atlas_to_image(atlas_file, gm_file, resampled_atlas)
        
        atlas_img = nib.load(resampled_atlas)
        atlas_data = atlas_img.get_fdata()
        region_labels = [int(i) for i in np.unique(atlas_data) if i != 0]
        
        atlas_labels_file = Path(config['atlas']['aal_labels'])
        label_map = pd.read_csv(
            atlas_labels_file, 
            sep=r'\s+', 
            header=None, 
            names=['label_name', 'long_name', 'label_id']
        )
        label_map = label_map.set_index('label_id')['label_name'].to_dict()
        
        print(f"  Extracting GM/WM/CSF volumes...")
        gm_volumes = extract_volumes(gm_file, resampled_atlas, region_labels, temp_dir)
        wm_volumes = extract_volumes(wm_file, resampled_atlas, region_labels, temp_dir)
        csf_volumes = extract_volumes(csf_file, resampled_atlas, region_labels, temp_dir)
        
        all_rows = []
        
        pet_columns = {f'PET_suvr_{label_map.get(label_id, f"Region_{label_id}")}': None 
                      for label_id in region_labels}
        
        if not pet_scan_ids or len(pet_scan_ids) == 0:
            print(f"  No PET scans found, creating MRI-only features...")
            
            row = {'subject_id': mri_subject_id, 'has_pet': False}
            
            for label_id in region_labels:
                region_name = label_map.get(label_id, f'Region_{label_id}')
                row[f'GM_vol_{region_name}'] = gm_volumes.get(label_id)
                row[f'WM_vol_{region_name}'] = wm_volumes.get(label_id)
                row[f'CSF_vol_{region_name}'] = csf_volumes.get(label_id)
            
            row.update(pet_columns)
            all_rows.append(row)
        else:
            for pet_scan_id in pet_scan_ids:
                print(f"  Processing PET scan: {pet_scan_id}")
                
                pet_coreg_dir = processed_dir / 'pet_coregistered'
                pet_file = pet_coreg_dir / f'{pet_scan_id}_coreg.nii.gz'
                
                row = {'subject_id': pet_scan_id, 'mri_subject_id': mri_subject_id}
                
                if not pet_file.exists():
                    print(f"    WARNING: Co-registered PET not found, MRI-only row")
                    row['has_pet'] = False
                    
                    for label_id in region_labels:
                        region_name = label_map.get(label_id, f'Region_{label_id}')
                        row[f'GM_vol_{region_name}'] = gm_volumes.get(label_id)
                        row[f'WM_vol_{region_name}'] = wm_volumes.get(label_id)
                        row[f'CSF_vol_{region_name}'] = csf_volumes.get(label_id)
                    
                    row.update(pet_columns)
                    all_rows.append(row)
                    continue
                
                row['has_pet'] = True
                
                pet_img = nib.load(pet_file)
                if pet_img.shape[:3] != atlas_img.shape[:3]:
                    print(f"    PET has different dimensions, resampling atlas...")
                    resampled_atlas_pet = temp_dir / f'atlas_resampled_{pet_scan_id}.nii.gz'
                    resample_atlas_to_image(atlas_file, pet_file, resampled_atlas_pet)
                    pet_suvrs = extract_suvr(
                        pet_file, 
                        resampled_atlas_pet, 
                        region_labels, 
                        config['atlas']['cerebellum_labels'],
                        temp_dir
                    )
                else:
                    pet_suvrs = extract_suvr(
                        pet_file, 
                        resampled_atlas, 
                        region_labels, 
                        config['atlas']['cerebellum_labels'],
                        temp_dir
                    )
                
                for label_id in region_labels:
                    region_name = label_map.get(label_id, f'Region_{label_id}')
                    row[f'GM_vol_{region_name}'] = gm_volumes.get(label_id)
                    row[f'WM_vol_{region_name}'] = wm_volumes.get(label_id)
                    row[f'CSF_vol_{region_name}'] = csf_volumes.get(label_id)
                    if pet_suvrs:
                        row[f'PET_suvr_{region_name}'] = pet_suvrs.get(label_id)
                
                all_rows.append(row)
        
        result['features_df'] = pd.DataFrame(all_rows)
        
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        result['status'] = 'success'
        return result
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"  ✗ ERROR: {e}")
        traceback.print_exc()
        return result

if __name__ == '__main__':
    print("--- Starting Single Subject Feature Extraction ---")
    
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'processing_config.yaml'
    config = load_config(config_path)
    
    # example
    mri_subject_id = '002_S_1155_MPRAGE_20111222_I274154'
    pet_scan_ids = [
        '002_S_1155_ADNI_Brain_PET__Raw_FDG_20110107_I209934',
        '002_S_1155_ADNI_Brain_PET__Raw_FDG_20130103_I353256'
    ]
    
    result = extract_features_single_subject(mri_subject_id, pet_scan_ids, config)
    
    if result['status'] == 'success':
        print(f"\n✓ Feature extraction completed!")
        print(f"  Output files: {len(result['output_files'])}")
    else:
        print(f"\n✗ Feature extraction failed: {result['error']}")