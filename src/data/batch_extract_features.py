from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import yaml
import time
from datetime import datetime
from filelock import FileLock
import extract_features

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_progress_tracker(log_dir, subjects):
    """Initialize or load progress tracking CSV"""
    tracker_file = log_dir / 'feature_extraction_status.csv'
    
    if tracker_file.exists():
        print(f"Loading existing progress tracker: {tracker_file}")
        df = pd.read_csv(tracker_file, dtype=str)
        
        existing_ids = set(df['mri_subject_id'].tolist())
        new_subjects = [s for s in subjects if s['mri_subject_id'] not in existing_ids]
        
        if new_subjects:
            print(f"Adding {len(new_subjects)} new subjects to tracker...")
            new_rows = pd.DataFrame({
                'mri_subject_id': [s['mri_subject_id'] for s in new_subjects],
                'pet_scan_ids': [','.join(s['pet_scan_ids']) for s in new_subjects],
                'status': ['pending'] * len(new_subjects),
                'features_extracted': ['0'] * len(new_subjects),
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
            'mri_subject_id': [s['mri_subject_id'] for s in subjects],
            'pet_scan_ids': [','.join(s['pet_scan_ids']) for s in subjects],
            'status': ['pending'] * len(subjects),
            'features_extracted': ['0'] * len(subjects),
            'last_attempt': [''] * len(subjects),
            'error_message': [''] * len(subjects),
            'attempts': ['0'] * len(subjects),
            'processing_time_sec': ['0.0'] * len(subjects)
        })
        df.to_csv(tracker_file, index=False)
    
    return tracker_file

def update_progress(tracker_file, mri_subject_id, updates):
    """Update progress for a subject with file locking for multiprocessing safety"""
    lock_file = str(tracker_file) + '.lock'
    lock = FileLock(lock_file, timeout=30)
    
    try:
        with lock:
            df = pd.read_csv(tracker_file, dtype=str)
            idx = df[df['mri_subject_id'] == mri_subject_id].index
            
            if len(idx) > 0:
                for key, value in updates.items():
                    df.loc[idx[0], key] = str(value)
                df.loc[idx[0], 'last_attempt'] = datetime.now().isoformat()
                df.to_csv(tracker_file, index=False)
    except filelock.Timeout:
        print(f"  Warning: Could not acquire lock for {mri_subject_id}, skipping progress update")

def process_subject_wrapper(args):
    """Wrapper for parallel processing"""
    subject, config, tracker_file = args
    
    start_time = time.time()
    
    try:
        lock_file = str(tracker_file) + '.lock'
        lock = FileLock(lock_file, timeout=30)
        
        current_attempts = 0
        try:
            with lock:
                df = pd.read_csv(tracker_file, dtype=str)
                idx = df[df['mri_subject_id'] == subject['mri_subject_id']].index
                if len(idx) > 0:
                    current_attempts = int(df.loc[idx[0], 'attempts'])
        except:
            pass
        
        update_progress(tracker_file, subject['mri_subject_id'], {
            'status': 'processing',
            'attempts': current_attempts + 1
        })
        
        result = extract_features.extract_features_single_subject(
            subject['mri_subject_id'],
            subject['pet_scan_ids'],
            config,
            skip_existing=False
        )
        
        processing_time = time.time() - start_time
        
        num_rows = len(result['features_df']) if result['features_df'] is not None else 0
        
        update_progress(tracker_file, subject['mri_subject_id'], {
            'status': result['status'],
            'features_extracted': num_rows,
            'error_message': result.get('error', ''),
            'processing_time_sec': f'{processing_time:.2f}'
        })
        
        result['num_rows'] = num_rows
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        update_progress(tracker_file, subject['mri_subject_id'], {
            'status': 'failed',
            'error_message': str(e),
            'processing_time_sec': f'{processing_time:.2f}'
        })
        return {'mri_subject_id': subject['mri_subject_id'], 'status': 'failed', 'error': str(e)}

def discover_subjects_for_feature_extraction(config):
    """
    Discovers subjects ready for feature extraction by checking actual file existence.
    Does not rely on potentially outdated log status, only verifies files exist.
    """
    processed_dir = Path(config['paths']['processed'])
    
    print("  Scanning segmentation directories...")
    seg_base_dir = processed_dir / 'segmentation'
    
    successful_mris = []
    if seg_base_dir.exists():
        for seg_dir in seg_base_dir.iterdir():
            if seg_dir.is_dir():
                mri_id = seg_dir.name
                
                gm_files = list(seg_dir.glob(f'wc1*{mri_id}*.nii'))
                wm_files = list(seg_dir.glob(f'wc2*{mri_id}*.nii'))
                csf_files = list(seg_dir.glob(f'wc3*{mri_id}*.nii'))
                
                if gm_files and wm_files and csf_files:
                    successful_mris.append(mri_id)
    
    print(f"  Found {len(successful_mris)} MRI subjects with valid segmentation")
    
    # get successfully co-registered pet scans grouped by mri
    mri_to_pet_map = {mri_id: [] for mri_id in successful_mris}
    
    pet_log_path = Path(config['paths']['logs']) / 'pet_processing_status.csv'
    if pet_log_path.exists():
        try:
            pet_df = pd.read_csv(pet_log_path, dtype=str, on_bad_lines='skip')
            
            for _, row in pet_df.iterrows():
                if 'pet_scan_id' not in row or 'mri_subject_id' not in row:
                    continue
                    
                pet_scan_id = row['pet_scan_id']
                mri_subject_id = row['mri_subject_id']
                status = row.get('status', '')
                
                if status != 'success':
                    continue
                
                pet_file = processed_dir / 'pet_coregistered' / f'{pet_scan_id}_coreg.nii.gz'
                
                if mri_subject_id in mri_to_pet_map and pet_file.exists():
                    mri_to_pet_map[mri_subject_id].append(pet_scan_id)
        except Exception as e:
            print(f"  Warning: Error reading PET log: {e}")
            print("  Will proceed with MRI-only features.")
        
        total_pets = sum(len(pets) for pets in mri_to_pet_map.values())
        print(f"  Found {total_pets} successfully co-registered PET scans")
    else:
        print("  Note: No PET processing log found, will extract MRI-only features.")
    
    # create subject list
    subjects = []
    for mri_id in successful_mris:
        subjects.append({
            'mri_subject_id': mri_id,
            'pet_scan_ids': mri_to_pet_map[mri_id]
        })
    
    return subjects

def print_progress_summary(tracker_file, config):
    """Print summary of progress"""
    df = pd.read_csv(tracker_file, dtype=str)
    
    total = len(df)
    completed = len(df[df['status'] == 'success'])
    failed = len(df[df['status'] == 'failed'])
    pending = len(df[df['status'] == 'pending'])
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total MRI subjects:   {total}")
    print(f"Completed:            {completed} ({completed/total*100:.1f}%)")
    print(f"Failed:               {failed} ({failed/total*100:.1f}%)")
    print(f"Pending:              {pending} ({pending/total*100:.1f}%)")
    
    if completed > 0:
        completed_df = df[df['status'] == 'success']
        total_features = completed_df['features_extracted'].astype(int).sum()
        print(f"Total feature files:  {total_features}")
        
        times = completed_df['processing_time_sec'].astype(float)
        avg_time = times.mean()
        remaining = pending
        n_workers = config['processing']['n_workers']
        est_time_hours = (remaining * avg_time) / 3600 / n_workers
        print(f"Avg time/subject:     {avg_time:.1f} sec")
        print(f"Estimated remaining:  {est_time_hours:.1f} hours")
    
    print("="*60)

def run_batch_processing(config):
    """Main batch processing function"""
    
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    features_dir = Path(config['paths']['features'])
    features_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nDiscovering subjects ready for feature extraction...")
    all_subjects = discover_subjects_for_feature_extraction(config)
    
    # count subjects with and without pet
    subjects_with_pet = sum(1 for s in all_subjects if len(s['pet_scan_ids']) > 0)
    subjects_without_pet = len(all_subjects) - subjects_with_pet
    
    print(f"✓ Found {len(all_subjects)} MRI subjects:")
    print(f"  - {subjects_with_pet} with PET data")
    print(f"  - {subjects_without_pet} MRI-only")
    
    if len(all_subjects) == 0:
        print("No subjects ready for feature extraction.")
        return
    
    tracker_file = init_progress_tracker(log_dir, all_subjects)
    
    df = pd.read_csv(tracker_file, dtype=str)
    
    if config['processing']['skip_existing']:
        pending_ids = df[df['status'].isin(['pending', 'failed'])]['mri_subject_id'].tolist()
        subjects = [s for s in all_subjects if s['mri_subject_id'] in pending_ids]
        completed_count = len(all_subjects) - len(subjects)
        print(f"✓ {completed_count} already completed, {len(subjects)} remaining.")
    else:
        subjects = all_subjects
    
    if len(subjects) == 0:
        print("\nAll subjects already processed!")
        print_progress_summary(tracker_file, config)
        return
    
    n_workers = config['processing']['n_workers']
    checkpoint_interval = config['processing']['checkpoint_interval']
    
    print(f"\n{'='*60}")
    print(f"Starting feature extraction:")
    print(f"  Workers: {n_workers}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    print(f"  Total to process: {len(subjects)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    incremental_file = features_dir / 'features_incremental.csv'
    incremental_lock = features_dir / 'features_incremental.csv.lock'
    
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
        
        batch_dataframes = []
        for result in results:
            if result['status'] == 'success' and result.get('features_df') is not None:
                batch_dataframes.append(result['features_df'])
        
        # save batch incrementally with file lock for concurrent access
        if batch_dataframes:
            batch_df = pd.concat(batch_dataframes, ignore_index=True)
            
            with FileLock(str(incremental_lock), timeout=300):
                file_exists = incremental_file.exists()
                if file_exists:
                    batch_df.to_csv(incremental_file, mode='a', header=False, index=False)
                else:
                    batch_df.to_csv(incremental_file, mode='w', header=True, index=False)
            
            print(f"  → Saved {len(batch_df)} features to {incremental_file.name}")
        
        batch_success = sum(1 for r in results if r['status'] == 'success')
        batch_failed = sum(1 for r in results if r['status'] == 'failed')
        
        print(f"\nBatch {batch_num} completed in {batch_time/60:.1f} minutes:")
        print(f"  ✓ Success: {batch_success}")
        print(f"  ✗ Failed:  {batch_failed}")
        
        print_progress_summary(tracker_file, config)
    
    total_time = time.time() - start_time
    
    if incremental_file.exists():
        print(f"\n{'='*60}")
        print("CREATING FINAL FEATURE FILES")
        print(f"{'='*60}")
        
        consolidated_df = pd.read_csv(incremental_file, low_memory=False)
        
        if 'has_pet' in consolidated_df.columns:
            consolidated_df['has_pet'] = consolidated_df['has_pet'].map({'True': True, 'False': False})
            consolidated_df['has_pet'] = consolidated_df['has_pet'].fillna(False)
        
        # save consolidated dataset
        output_file = features_dir / 'all_features.csv'
        consolidated_df.to_csv(output_file, index=False)
        
        print(f"✓ Consolidated dataset created:")
        print(f"  File: {output_file}")
        print(f"  Samples: {len(consolidated_df)}")
        print(f"  Features: {len(consolidated_df.columns)}")
        if 'has_pet' in consolidated_df.columns:
            print(f"  With PET: {consolidated_df['has_pet'].sum()}")
        else:
            print(f"  With PET: N/A")
        
        if 'has_pet' in consolidated_df.columns:
            pet_df = consolidated_df[consolidated_df['has_pet']]
            if len(pet_df) > 0:
                pet_file = features_dir / 'features_with_pet.csv'
                pet_df.to_csv(pet_file, index=False)
                print(f"\n✓ MRI+PET dataset: {pet_file} ({len(pet_df)} samples)")
            
            mri_only_df = consolidated_df[~consolidated_df['has_pet']]
            if len(mri_only_df) > 0:
                mri_file = features_dir / 'features_mri_only.csv'
                mri_only_df.to_csv(mri_file, index=False)
                print(f"✓ MRI-only dataset: {mri_file} ({len(mri_only_df)} samples)")
    
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION COMPLETED")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")
    print_progress_summary(tracker_file, config)

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    
    print("="*60)
    print("BATCH FEATURE EXTRACTION PIPELINE")
    print(f"Project Root: {project_root}")
    print("="*60)
    
    config_path = project_root / 'configs' / 'processing_config.yaml'
    
    try:
        config = load_config(config_path)
        
        for key, value in config['paths'].items():
            config['paths'][key] = str(project_root / value)
            
        config['atlas']['aal_file'] = str(project_root / config['atlas']['aal_file'])
        config['atlas']['aal_labels'] = str(project_root / config['atlas']['aal_labels'])

    except FileNotFoundError:
        print(f"ERROR: Config file not found: {config_path}")
        exit(1)
    
    run_batch_processing(config)
