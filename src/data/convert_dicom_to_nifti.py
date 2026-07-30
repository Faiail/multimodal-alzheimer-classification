import subprocess
from pathlib import Path
import os
import traceback
import yaml
import pandas as pd
from datetime import datetime
from discover_data import get_scan_sessions

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def convert_dicom_to_nifti(dicom_dir, output_dir, subject_id):
    """
    Convert a DICOM directory to NIfTI format.
    
    Args:
        dicom_dir: Path to DICOM files
        output_dir: Output directory for NIfTI
        subject_id: Subject identifier for consistent naming
    
    Returns:
        Path to created NIfTI file or None if failed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    existing_files = set(output_dir.glob("*.nii.gz"))
    
    command = [
        'dcm2niix',
        '-o', str(output_dir),
        '-z', 'y',
        '-f', subject_id,
        '-b', 'n',
        str(dicom_dir)
    ]
    
    try:
        if not dicom_dir.is_dir():
            raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
        
        dcm_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        new_files = set(output_dir.glob("*.nii.gz")) - existing_files
        
        if new_files:
            matching = [f for f in new_files if subject_id in f.name]
            if matching:
                return matching[0]
            else:
                return list(new_files)[0]
        else:
            expected_file = output_dir / f"{subject_id}.nii.gz"
            if expected_file.exists():
                print(f"  ℹ File already exists (skipped by dcm2niix)")
                return expected_file
            
            raise FileNotFoundError(f"No NIfTI file created for {subject_id}")
            
    except subprocess.CalledProcessError as e:
        print(f"  ✗ dcm2niix error: {e.stderr}")
        return None
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return None

def init_conversion_tracker(log_dir, subjects):
    """Initialize or load conversion tracking CSV"""
    tracker_file = log_dir / 'dicom_conversion_status.csv'
    
    if tracker_file.exists():
        print(f"Loading existing conversion tracker: {tracker_file}")
        df = pd.read_csv(tracker_file, dtype=str)
    else:
        print(f"Creating new conversion tracker: {tracker_file}")
        df = pd.DataFrame({
            'subject_id': [s['id'] for s in subjects],
            'subject_folder': [s['subject_folder'] for s in subjects],
            'scan_type': [s['scan_type'] for s in subjects],
            'modality': [s['modality'] for s in subjects],
            'dicom_dir': [str(s['dicom_dir']) for s in subjects],
            'status': ['pending'] * len(subjects),
            'output_file': ['' for _ in subjects],
            'conversion_time': [0.0 for _ in subjects],
            'error_message': ['' for _ in subjects],
            'timestamp': ['' for _ in subjects]
        })
        df.to_csv(tracker_file, index=False)
    
    return tracker_file

def update_conversion_tracker(tracker_file, subject_id, status, output_file=None, error=None, conversion_time=None):
    """Update conversion tracker"""
    df = pd.read_csv(tracker_file, dtype=str)
    idx = df[df['subject_id'] == subject_id].index
    
    if len(idx) > 0:
        df.loc[idx[0], 'status'] = status
        df.loc[idx[0], 'timestamp'] = datetime.now().isoformat()
        if output_file:
            df.loc[idx[0], 'output_file'] = str(output_file)
        if error:
            df.loc[idx[0], 'error_message'] = str(error)
        if conversion_time is not None:
            df.loc[idx[0], 'conversion_time'] = f'{conversion_time:.2f}'
        df.to_csv(tracker_file, index=False)

def print_conversion_summary(tracker_file):
    """Print summary of conversion progress"""
    df = pd.read_csv(tracker_file)
    
    total = len(df)
    completed = len(df[df['status'] == 'success'])
    failed = len(df[df['status'] == 'failed'])
    pending = len(df[df['status'] == 'pending'])
    
    print("\n" + "="*60)
    print("DICOM CONVERSION SUMMARY")
    print("="*60)
    print(f"Total subjects:     {total}")
    print(f"Completed:          {completed} ({completed/total*100:.1f}%)")
    print(f"Failed:             {failed} ({failed/total*100:.1f}%)")
    print(f"Pending:            {pending} ({pending/total*100:.1f}%)")
    
    print("\nBy Modality:")
    for modality in df['modality'].unique():
        count = len(df[df['modality'] == modality])
        print(f"  {modality}: {count}")
    
    print("="*60)

if __name__ == '__main__':
    import time
    
    print("="*60)
    print("DICOM TO NIFTI BATCH CONVERSION")
    print("="*60)
    
    config_path = 'configs/processing_config.yaml'
    config = load_config(config_path)
    
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config['paths']['processed']) / 'nifti'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Discovering DICOM subjects...")
    mri_tasks, pet_tasks = get_scan_sessions(config)
    
    subjects = []
    for task in mri_tasks:
        subjects.append({
            'id': task['subject_id'],
            'subject_folder': task['subject_id'].split('_')[0],
            'scan_type': 'MRI',
            'modality': task['modality'],
            'dicom_dir': Path(task['dicom_path']),
        })
    for task in pet_tasks:
        subjects.append({
            'id': task['pet_scan_id'],
            'subject_folder': task['subject_id'],
            'scan_type': 'PET',
            'modality': task['modality'],
            'dicom_dir': Path(task['dicom_path']),
        })

    print(f"✓ Found {len(subjects)} DICOM series ({len(mri_tasks)} MRI, {len(pet_tasks)} PET)")
    
    if len(subjects) == 0:
        print("No DICOM subjects found. Exiting.")
        exit(0)
    
    print("\n[2/3] Initializing conversion tracker...")
    tracker_file = init_conversion_tracker(log_dir, subjects)
    
    if config['processing']['skip_existing']:
        df = pd.read_csv(tracker_file)
        pending_ids = df[df['status'].isin(['pending', 'failed'])]['subject_id'].tolist()
        subjects = [s for s in subjects if s['id'] in pending_ids]
        print(f"✓ Skipping completed subjects. {len(subjects)} remaining to convert.")
    
    if len(subjects) == 0:
        print("\nAll subjects already converted!")
        print_conversion_summary(tracker_file)
        exit(0)
    
    print(f"\n[3/3] Converting {len(subjects)} subjects...")
    print("-" * 60)
    
    start_time = time.time()
    
    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {subject['id']}")
        print(f"  Modality: {subject['modality']} | Scan: {subject['scan_type']}")
        print(f"  DICOM dir: .../{subject['dicom_dir'].name}")
        
        sub_start = time.time()
        
        nifti_file = convert_dicom_to_nifti(
            subject['dicom_dir'],
            output_dir,
            subject['id']
        )
        
        conversion_time = time.time() - sub_start
        
        if nifti_file:
            print(f"  ✓ Created: {nifti_file.name} ({conversion_time:.1f}s)")
            update_conversion_tracker(
                tracker_file, 
                subject['id'], 
                'success',
                output_file=nifti_file,
                conversion_time=conversion_time
            )
        else:
            print(f"  ✗ Failed ({conversion_time:.1f}s)")
            update_conversion_tracker(
                tracker_file,
                subject['id'],
                'failed',
                error='Conversion failed',
                conversion_time=conversion_time
            )
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"CONVERSION COMPLETED in {total_time/60:.1f} minutes")
    print("="*60)
    print_conversion_summary(tracker_file)
    
    print(f"\nConversion log saved to: {tracker_file}")
    print(f"NIfTI files saved to: {output_dir}")