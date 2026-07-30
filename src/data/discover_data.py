import pandas as pd
from pathlib import Path
import re

def find_dicom_directories(base_dir):
    """Find all directories containing DICOM files."""
    base_dir = Path(base_dir)
    return [p.parent for p in base_dir.rglob("*.dcm")]

def parse_path(path, base_dir):
    """
    Parses the path to extract subject, type, date, and scan_id.
    This function is customized for the user's specific directory structure.
    """
    base_dir = Path(base_dir)
    parts = path.relative_to(base_dir).parts

    if len(parts) < 4:
        return None

    subject_id = parts[0]
    scan_type = parts[1]
    date_str = parts[2]
    scan_id = parts[3]

    if not re.match(r'I\d+', scan_id):
        return None
    if not re.match(r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.\d', date_str):
        return None

    try:
        scan_date = pd.to_datetime(date_str.split('.')[0], format='%Y-%m-%d_%H_%M_%S')
    except ValueError:
        return None

    is_mri = 'mprage' in scan_type.lower()
    
    return {
        'subject_id': subject_id,
        'scan_type': 'MRI' if is_mri else 'PET',
        'modality': 'MPRAGE' if is_mri else scan_type,
        'scan_date': scan_date,
        'scan_id': scan_id,
        'dicom_path': str(path)
    }

def discover_scans(raw_data_dir):
    """
    Discovers all MRI and PET scans from the raw data directory, parsing
    information from the folder structure.
    """
    print("🔍 Starting custom discovery of DICOM data...")
    dicom_dirs = set(find_dicom_directories(raw_data_dir))
    
    if not dicom_dirs:
        print("  No DICOM directories found.")
        return pd.DataFrame()

    print(f"  Found {len(dicom_dirs)} directories with DICOM files. Parsing paths...")

    parsed_data = [parse_path(p, raw_data_dir) for p in dicom_dirs]
    scans_df = pd.DataFrame([d for d in parsed_data if d is not None])

    if scans_df.empty:
        print("  Could not parse any valid scan paths from the directory structure.")
        return pd.DataFrame()

    print(f"  Successfully parsed {len(scans_df)} scans.")
    return scans_df

def associate_pet_to_mri(scans_df):
    """
    Associates each PET scan to the closest-in-time MRI scan for the same subject.
    """
    if scans_df.empty or 'scan_type' not in scans_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    mri_df = scans_df[scans_df['scan_type'] == 'MRI'].copy()
    pet_df = scans_df[scans_df['scan_type'] == 'PET'].copy()

    if mri_df.empty:
        print("  Warning: No MRI scans found. Cannot associate PET scans.")
        return scans_df[scans_df['scan_type'] == 'MRI'], pd.DataFrame()
        
    if pet_df.empty:
        print("  No PET scans found to associate.")
        return mri_df, pd.DataFrame()

    print("🤝 Associating PET scans to the nearest-in-time MRI...")

    mri_df['mri_session_id'] = mri_df.apply(
        lambda row: f"{row['subject_id']}_{row['modality']}_{row['scan_id']}", axis=1
    )
    
    pet_df['mri_session_id'] = None
    pet_df['time_diff_days'] = None

    for idx, pet_row in pet_df.iterrows():
        subject = pet_row['subject_id']
        pet_date = pet_row['scan_date']
        
        subject_mris = mri_df[mri_df['subject_id'] == subject]
        
        if not subject_mris.empty:
            time_diffs = (subject_mris['scan_date'] - pet_date).abs()
            closest_mri_idx = time_diffs.idxmin()
            
            closest_mri = subject_mris.loc[closest_mri_idx]
            
            pet_df.loc[idx, 'mri_session_id'] = closest_mri['mri_session_id']
            pet_df.loc[idx, 'time_diff_days'] = time_diffs.min().days

    unassociated_pets = pet_df[pet_df['mri_session_id'].isnull()]
    if not unassociated_pets.empty:
        print(f"  Warning: Could not find a matching MRI for {len(unassociated_pets)} PET scans.")

    print("  Association complete.")
    return mri_df, pet_df[pet_df['mri_session_id'].notnull()]

def get_scan_sessions(config):
    """
    Main function to discover and associate scans based on the custom directory structure.
    """
    raw_data_dir = config['paths']['raw_dicom']
    
    all_scans_df = discover_scans(raw_data_dir)
    
    if all_scans_df.empty:
        return [], []
        
    mri_df, pet_df = associate_pet_to_mri(all_scans_df)
    
    mri_subjects = mri_df.to_dict('records')
    pet_subjects = pet_df.to_dict('records')
    
    pet_tasks = []
    for pet in pet_subjects:
        pet_tasks.append({
            'subject_id': pet['subject_id'],
            'dicom_path': pet['dicom_path'],
            'mri_session_id': pet['mri_session_id'],
            'modality': pet['modality'],
            'pet_scan_id': f"{pet['subject_id']}_{pet['modality']}_{pet['scan_id']}"
        })

    mri_tasks = []
    for mri in mri_subjects:
        mri_tasks.append({
            'subject_id': mri['mri_session_id'],
            'dicom_path': mri['dicom_path'],
            'modality': mri['modality']
        })

    return mri_tasks, pet_tasks

if __name__ == '__main__':
    import yaml
    
    dummy_config = {
        'paths': {
            'raw_dicom': str(Path(__file__).resolve().parents[2] / 'data' / '01_raw')
        }
    }
    
    base = Path(dummy_config['paths']['raw_dicom'])
    (base / 'Sub-01/MPRAGE/2022-01-15_10_30_00.0/I12345').mkdir(parents=True, exist_ok=True)
    (base / 'Sub-01/MPRAGE/2022-01-15_10_30_00.0/I12345/001.dcm').touch()
    (base / 'Sub-01/FDG_PET/2022-02-20_11_00_00.0/I67890').mkdir(parents=True, exist_ok=True)
    (base / 'Sub-01/FDG_PET/2022-02-20_11_00_00.0/I67890/001.dcm').touch()
    (base / 'Sub-02/MPRAGE/2023-05-01_09_00_00.0/I54321').mkdir(parents=True, exist_ok=True)
    (base / 'Sub-02/MPRAGE/2023-05-01_09_00_00.0/I54321/001.dcm').touch()

    mri_tasks, pet_tasks = get_scan_sessions(dummy_config)
    
    print("\n--- Discovered MRI Tasks ---")
    print(mri_tasks)
    
    print("\n--- Discovered PET Tasks ---")
    print(pet_tasks)
