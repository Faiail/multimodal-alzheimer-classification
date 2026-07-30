from multiprocessing import Pool, Manager
from pathlib import Path
import pandas as pd
import yaml
import time
from datetime import datetime
import re
import process_pet
import filelock

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_progress_tracker(log_dir, subjects):
    """Initialize or load progress tracking CSV"""
    tracker_file = log_dir / 'pet_processing_status.csv'
    
    if tracker_file.exists():
        print(f"Loading existing progress tracker: {tracker_file}")
        df = pd.read_csv(tracker_file, dtype=str)
        
        # use pet_scan_id as unique identifier (not subject_id which may...
        existing_ids = set(df['pet_scan_id'].tolist())
        new_subjects = [s for s in subjects if s['pet_scan_id'] not in existing_ids]
        
        if new_subjects:
            print(f"Adding {len(new_subjects)} new PET scans to tracker...")
            new_rows = pd.DataFrame({
                'pet_scan_id': [s['pet_scan_id'] for s in new_subjects],
                'mri_subject_id': [s['id'] for s in new_subjects],
                'pet_file': [str(s['pet_file']) for s in new_subjects],
                'reference_mri': [str(s['reference_mri']) for s in new_subjects],
                'status': ['pending'] * len(new_subjects),
                'output_file': [''] * len(new_subjects),
                'last_attempt': [''] * len(new_subjects),
                'error_message': [''] * len(new_subjects),
                'attempts': ['0'] * len(new_subjects),
                'processing_time_sec': ['0.0'] * len(new_subjects)
            })
            df = pd.concat([df, new_rows], ignore_index=True)
            df.to_csv(tracker_file, index=False)
    else:
        print(f"Creating new progress tracker: {tracker_file}")
        df = pd.DataFrame({
            'pet_scan_id': [s['pet_scan_id'] for s in subjects],
            'mri_subject_id': [s['id'] for s in subjects],
            'pet_file': [str(s['pet_file']) for s in subjects],
            'reference_mri': [str(s['reference_mri']) for s in subjects],
            'status': ['pending'] * len(subjects),
            'output_file': [''] * len(subjects),
            'last_attempt': [''] * len(subjects),
            'error_message': [''] * len(subjects),
            'attempts': ['0'] * len(subjects),
            'processing_time_sec': ['0.0'] * len(subjects)
        })
        df.to_csv(tracker_file, index=False)
    
    return tracker_file

def update_progress(tracker_file, pet_scan_id, updates):
    """Update progress for a PET scan with file locking for multiprocessing safety"""
    lock_file = str(tracker_file) + '.lock'
    lock = filelock.FileLock(lock_file, timeout=30)
    
    try:
        with lock:
            df = pd.read_csv(tracker_file, dtype=str)
            idx = df[df['pet_scan_id'] == pet_scan_id].index
            
            if len(idx) > 0:
                for key, value in updates.items():
                    df.loc[idx[0], key] = str(value)
                df.loc[idx[0], 'last_attempt'] = datetime.now().isoformat()
                df.to_csv(tracker_file, index=False)
    except filelock.Timeout:
        print(f"  Warning: Could not acquire lock for {pet_scan_id}, skipping progress update")

def process_subject_wrapper(args):
    """Wrapper for parallel processing"""
    subject, config, tracker_file = args
    
    start_time = time.time()
    
    try:
        lock_file = str(tracker_file) + '.lock'
        lock = filelock.FileLock(lock_file, timeout=30)
        
        current_attempts = 0
        try:
            with lock:
                df = pd.read_csv(tracker_file, dtype=str)
                idx = df[df['pet_scan_id'] == subject['pet_scan_id']].index
                if len(idx) > 0:
                    current_attempts = int(df.loc[idx[0], 'attempts'])
        except:
            pass
        
        update_progress(tracker_file, subject['pet_scan_id'], {
            'status': 'processing',
            'attempts': current_attempts + 1
        })
        
        result = process_pet.process_single_pet(
            subject['pet_file'],
            subject['reference_mri'],
            subject['id'],  # mri subject id
            subject['pet_scan_id'],  # unique pet scan id
            config,
            skip_existing=config['processing']['skip_existing']
        )
        
        processing_time = time.time() - start_time
        
        update_progress(tracker_file, subject['pet_scan_id'], {
            'status': result['status'],
            'output_file': str(result['output_file']) if result['output_file'] else '',
            'error_message': result.get('error', ''),
            'processing_time_sec': f'{processing_time:.2f}'  # format as string
        })
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        update_progress(tracker_file, subject['pet_scan_id'], {
            'status': 'failed',
            'error_message': str(e),
            'processing_time_sec': f'{processing_time:.2f}'
        })
        return {'pet_scan_id': subject['pet_scan_id'], 'status': 'failed', 'error': str(e)}

def extract_date_from_dicom_dir(dicom_dir):
    """Extract date from dicom_dir path like 'data/01_raw/.../2012-03-09_09_29_41.0/I289652'"""
    if pd.isna(dicom_dir):
        return None
    match = re.search(r'(\d{4}-\d{2}-\d{2})_\d{2}_\d{2}_\d{2}', str(dicom_dir))
    if match:
        try:
            return pd.to_datetime(match.group(1))
        except:
            return None
    return None

def match_pet_to_mri(config):
    """
    Identifies valid PET-MRI pairs for processing using existing logs (no DICOM discovery).
    
    1. Loads successful PET conversions from dicom_conversion_status.csv
    2. Loads MRI entries from mri_processing_status.csv and conversion log
    3. Matches PET to MRI by subject folder (e.g., 002_S_4521)
    4. For each PET, finds the temporally closest MRI
    5. Returns pairs where all necessary input files exist.
    """
    conversion_log_path = Path(config['paths']['logs']) / 'dicom_conversion_status.csv'
    mri_log_path = Path(config['paths']['logs']) / 'mri_processing_status.csv'

    if not conversion_log_path.exists():
        raise FileNotFoundError(f"Conversion log not found: {conversion_log_path}")
    if not mri_log_path.exists():
        raise FileNotFoundError(f"MRI processing log not found: {mri_log_path}")

    conversion_df = pd.read_csv(conversion_log_path)
    mri_df = pd.read_csv(mri_log_path)

    # filter successful pet scans
    pet_df = conversion_df[(conversion_df['status'] == 'success') & (conversion_df['scan_type'] == 'PET')].copy()
    
    # get mri entries from conversion log (has dicom_dir with dates)
    mri_conversion_df = conversion_df[(conversion_df['status'] == 'success') & (conversion_df['scan_type'] == 'MRI')].copy()
    
    print(f"  Found {len(pet_df)} successful PET conversions in log")
    print(f"  Found {len(mri_conversion_df)} successful MRI conversions in log")
    
    if len(pet_df) == 0:
        return []
    
    # extract dates from dicom_dir
    pet_df['scan_date'] = pet_df['dicom_dir'].apply(extract_date_from_dicom_dir)
    mri_conversion_df['scan_date'] = mri_conversion_df['dicom_dir'].apply(extract_date_from_dicom_dir)
    
    def extract_subject_folder(subject_id):
        match = re.match(r'^(\d{3}_S_\d{4})', subject_id)
        return match.group(1) if match else None
    
    pet_df['subject_folder'] = pet_df['subject_id'].apply(extract_subject_folder)
    mri_conversion_df['subject_folder'] = mri_conversion_df['subject_id'].apply(extract_subject_folder)
    
    # build lookup: subject_folder -> list of (mri_subject_id
    mri_by_subject = {}
    for _, row in mri_conversion_df.iterrows():
        sf = row['subject_folder']
        if sf is None:
            continue
        if sf not in mri_by_subject:
            mri_by_subject[sf] = []
        mri_by_subject[sf].append({
            'subject_id': row['subject_id'],
            'scan_date': row['scan_date']
        })
    
    processed_dir = Path(config['paths']['processed'])
    
    matched_subjects = []
    skipped_no_mri = 0
    skipped_no_file = 0
    skipped_no_date = 0
    
    for _, pet_row in pet_df.iterrows():
        pet_scan_id = pet_row['subject_id']
        subject_folder = pet_row['subject_folder']
        pet_date = pet_row['scan_date']
        
        if subject_folder is None:
            continue
            
        # check if we have any mri for this subject
        if subject_folder not in mri_by_subject:
            skipped_no_mri += 1
            continue
        
        # build pet file path
        pet_output = pet_row['output_file']
        # handle relative vs absolute paths
        if pet_output.startswith('data/'):
            pet_file = Path(config['paths']['processed']).parent.parent / pet_output
        else:
            pet_file = Path(pet_output)
        
        # check if pet file exists
        if not pet_file.exists():
            skipped_no_file += 1
            continue
        
        # find the temporally closest mri with existing n4 file
        mri_candidates = mri_by_subject[subject_folder]
        best_mri = None
        best_n4_file = None
        min_time_diff = None
        
        for mri_info in mri_candidates:
            mri_subject_id = mri_info['subject_id']
            mri_date = mri_info['scan_date']
            
            # check if n4 file exists
            candidate_n4 = processed_dir / 'n4' / f'{mri_subject_id}_n4.nii.gz'
            if not candidate_n4.exists():
                continue
            
            # calculate time difference (if dates available)
            if pet_date is not None and mri_date is not None:
                time_diff = abs((pet_date - mri_date).days)
                if min_time_diff is None or time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_mri = mri_subject_id
                    best_n4_file = candidate_n4
            elif best_mri is None:
                best_mri = mri_subject_id
                best_n4_file = candidate_n4
        
        if best_mri is None:
            skipped_no_file += 1
            continue
        
        matched_subjects.append({
            'id': best_mri,
            'pet_scan_id': pet_scan_id,
            'pet_file': pet_file,
            'reference_mri': best_n4_file,
            'subject_folder': subject_folder,
            'time_diff_days': min_time_diff
        })
    
    print(f"  Skipped {skipped_no_mri} PET scans without matching MRI")
    print(f"  Skipped {skipped_no_file} pairs due to missing files")
    print(f"\n✓ Found {len(matched_subjects)} valid PET-MRI pairs ready for processing.")
    
    return matched_subjects

def print_progress_summary(tracker_file, config):
    """Print summary of progress"""
    df = pd.read_csv(tracker_file, dtype=str)
    
    total = len(df)
    completed = len(df[df['status'] == 'success'])
    failed = len(df[df['status'] == 'failed'])
    pending = len(df[df['status'] == 'pending'])
    processing = len(df[df['status'] == 'processing'])
    
    print("\n" + "="*60)
    print("PET PROCESSING SUMMARY")
    print("="*60)
    print(f"Total subjects:       {total}")
    print(f"Completed:            {completed} ({completed/total*100:.1f}%)")
    print(f"Failed:               {failed} ({failed/total*100:.1f}%)")
    print(f"Pending:              {pending} ({pending/total*100:.1f}%)")
    print(f"Currently processing: {processing}")
    print("="*60)
    
    if completed > 0:
        completed_df = df[df['status'] == 'success']
        # convert to float for calculation
        times = completed_df['processing_time_sec'].astype(float)
        avg_time = times.mean()
        remaining = pending + processing
        n_workers = config['processing']['n_workers']
        est_time_hours = (remaining * avg_time) / 3600 / n_workers
        print(f"Avg processing time:  {avg_time:.1f} sec/subject")
        print(f"Estimated remaining:  {est_time_hours:.1f} hours ({est_time_hours*60:.0f} min)")
        print("="*60)
    
    if failed > 0:
        print("\nFailed subjects:")
        failed_df = df[df['status'] == 'failed'][['pet_scan_id', 'error_message']]
        for _, row in failed_df.head(5).iterrows():
            error_short = row['error_message'][:60] + '...' if len(row['error_message']) > 60 else row['error_message']
            print(f"  - {row['pet_scan_id'][:40]}...: {error_short}")
        if failed > 5:
            print(f"  ... and {failed - 5} more")
        print("="*60)

def run_batch_processing(config):
    """Main batch processing function"""
    
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # match pet scans to mri scans
    print("\nMatching PET scans to processed MRI scans...")
    all_pet_mri_pairs = match_pet_to_mri(config)
    print(f"✓ Found {len(all_pet_mri_pairs)} PET-MRI pairs total")
    
    if len(all_pet_mri_pairs) == 0:
        print("No PET-MRI pairs found. Exiting.")
        return
    
    # initialize progress tracker (will add new subjects if needed)
    print("\nInitializing progress tracker...")
    tracker_file = init_progress_tracker(log_dir, all_pet_mri_pairs)
    
    df = pd.read_csv(tracker_file, dtype=str)
    
    # get ids of subjects discovered in this run
    discovered_ids = [s['pet_scan_id'] for s in all_pet_mri_pairs]  # use pet_scan_id
    
    # debug: check actual file existence vs tracker status
    print("\nDEBUG: Checking actual file existence...")
    pet_coreg_dir = Path(config['paths']['processed']) / 'pet_coregistered'
    for pair in all_pet_mri_pairs:
        expected_file = pet_coreg_dir / f"{pair['pet_scan_id']}_coreg.nii.gz"
        exists = expected_file.exists()
        tracker_row = df[df['pet_scan_id'] == pair['pet_scan_id']]
        if len(tracker_row) > 0:
            tracker_status = tracker_row.iloc[0]['status']
            print(f"  {pair['pet_scan_id'][:50]}... | File exists: {exists} | Tracker: {tracker_status}")
        else:
            print(f"  {pair['pet_scan_id'][:50]}... | File exists: {exists} | Tracker: NOT IN TRACKER")
    
    # filter to discovered subjects that are pending or failed
    if config['processing']['skip_existing']:
        # check actual file existence, not just tracker status
        subjects_to_process = []
        for pair in all_pet_mri_pairs:
            expected_file = pet_coreg_dir / f"{pair['pet_scan_id']}_coreg.nii.gz"
            
            # get tracker status
            tracker_row = df[df['pet_scan_id'] == pair['pet_scan_id']]
            if len(tracker_row) > 0:
                tracker_status = tracker_row.iloc[0]['status']
            else:
                tracker_status = 'pending'
            
            if not expected_file.exists() or tracker_status in ['pending', 'failed']:
                subjects_to_process.append(pair)
        
        subjects = subjects_to_process
        completed_count = len(all_pet_mri_pairs) - len(subjects)
        print(f"\n✓ {completed_count} already completed, {len(subjects)} PET scans remaining to process.")
    else:
        subjects = all_pet_mri_pairs
        print(f"✓ Processing all {len(subjects)} PET scans (skip_existing=False)")
    
    if len(subjects) == 0:
        print("\nAll PET subjects already processed!")
        print_progress_summary(tracker_file, config)
        return
    
    n_workers = config['processing']['n_workers']
    checkpoint_interval = config['processing']['checkpoint_interval']
    
    print(f"\n{'='*60}")
    print(f"Starting parallel processing:")
    print(f"  Workers: {n_workers}")
    print(f"  Checkpoint interval: every {checkpoint_interval} subjects")
    print(f"  Total to process: {len(subjects)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for batch_start in range(0, len(subjects), checkpoint_interval):
        batch_end = min(batch_start + checkpoint_interval, len(subjects))
        batch = subjects[batch_start:batch_end]
        
        batch_num = batch_start // checkpoint_interval + 1
        total_batches = (len(subjects) + checkpoint_interval - 1) // checkpoint_interval
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches}: Processing subjects {batch_start+1} to {batch_end}")
        print(f"{'='*60}")
        
        args = [(s, config, tracker_file) for s in batch]
        
        batch_start_time = time.time()
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_subject_wrapper, args)
        batch_time = time.time() - batch_start_time
        
        batch_success = sum(1 for r in results if r['status'] == 'success')
        batch_failed = sum(1 for r in results if r['status'] == 'failed')
        
        print(f"\nBatch {batch_num} completed in {batch_time/60:.1f} minutes:")
        print(f"  ✓ Success: {batch_success}")
        print(f"  ✗ Failed:  {batch_failed}")
        print(f"  Checkpoint saved to: {tracker_file}")
        
        print_progress_summary(tracker_file, config)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETED")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")
    print_progress_summary(tracker_file, config)

if __name__ == '__main__':
    print("="*60)
    print("BATCH PET CO-REGISTRATION PIPELINE")
    print("="*60)
    
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'processing_config.yaml'
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {config_path}")
        print("Please create the configuration file first.")
        exit(1)
    
    run_batch_processing(config)
