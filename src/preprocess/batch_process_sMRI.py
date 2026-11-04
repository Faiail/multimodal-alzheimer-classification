from multiprocessing import Pool, Lock
from pathlib import Path
import pandas as pd
import yaml
import time
from datetime import datetime
import os
from src.preprocess import process_sMRI
from src.utils.general import resolve_data_path

csv_lock = Lock()

def load_config(config_path):
    """Load config and resolve all relative paths to absolute paths"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    workspace_root = Path(config_path).resolve().parent.parent
    
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str) and not os.path.isabs(value):
                config['paths'][key] = str(workspace_root / value)
    
    return config

def init_progress_tracker(log_dir, subjects):
    """Initialize or load progress tracking CSV"""
    tracker_file = log_dir / 'mri_processing_status.csv'
    
    if tracker_file.exists():
        print(f"Loading existing progress tracker: {tracker_file}")
        df = pd.read_csv(tracker_file, dtype=str)
    else:
        print(f"Creating new progress tracker: {tracker_file}")
        df = pd.DataFrame({
            'subject_id': [s['id'] for s in subjects],
            'input_file': [str(s['input_file']) for s in subjects],
            'status': ['pending'] * len(subjects),
            'n4_completed': ['False'] * len(subjects),
            'segmentation_completed': ['False'] * len(subjects),
            'last_attempt': [''] * len(subjects),
            'error_message': [''] * len(subjects),
            'attempts': ['0'] * len(subjects),
            'processing_time_sec': ['0.0'] * len(subjects)
        })
        df.to_csv(tracker_file, index=False)
    
    return tracker_file

def update_progress(tracker_file, subject_id, updates):
    """Update progress for a subject (thread-safe)"""
    with csv_lock:
        df = pd.read_csv(tracker_file, dtype=str)
        idx = df[df['subject_id'] == subject_id].index
        
        if len(idx) > 0:
            for key, value in updates.items():
                df.loc[idx[0], key] = str(value)
            df.loc[idx[0], 'last_attempt'] = datetime.now().isoformat()
            df.to_csv(tracker_file, index=False)

def process_subject_wrapper(args):
    """Wrapper for parallel processing"""
    subject, config, tracker_file = args
    
    start_time = time.time()
    
    try:
        df = pd.read_csv(tracker_file)
        idx = df[df['subject_id'] == subject['id']].index
        current_attempts = int(df.loc[idx[0], 'attempts']) if len(idx) > 0 else 0
        
        update_progress(tracker_file, subject['id'], {
            'status': 'processing',
            'attempts': current_attempts + 1
        })
        
        result = process_sMRI.process_single_subject(
            subject['input_file'],
            subject['id'],
            config,
            skip_existing=config['processing']['skip_existing']
        )

        processing_time = time.time() - start_time
        
        update_progress(tracker_file, subject['id'], {
            'status': result['status'],
            'n4_completed': result['n4_file'] is not None,
            'segmentation_completed': bool(result['segmentation_files']),
            'error_message': result.get('error', ''),
            'processing_time_sec': processing_time
        })
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        update_progress(tracker_file, subject['id'], {
            'status': 'failed',
            'error_message': str(e),
            'processing_time_sec': processing_time
        })
        return {'subject_id': subject['id'], 'status': 'failed', 'error': str(e)}

def discover_mri_subjects(config):
    """
    Discover MRI subjects from conversion log (optimized - no re-discovery).
    """
    conversion_log = Path(config['paths']['logs']) / 'dicom_conversion_status.csv'
    if not conversion_log.exists():
        raise FileNotFoundError(
            f"Conversion log not found: {conversion_log}\n"
            "Please run convert_dicom_to_nifti.py first."
        )
    
    print(f"  Reading from conversion log: {conversion_log}")
    log_df = pd.read_csv(conversion_log)
    
    successful_mri_df = log_df[(log_df['status'] == 'success') & (log_df['scan_type'] == 'MRI')]
    print(f"  Found {len(successful_mri_df)} successfully converted MRI subjects.")
    
    subjects = []
    workspace_root = Path(config['paths']['processed']).parent.parent
    
    for _, row in successful_mri_df.iterrows():
        subject_id = row['subject_id']
        output_file_rel = Path(row['output_file'])
        if output_file_rel.is_absolute():
            output_file = output_file_rel
        else:
            output_file = workspace_root / output_file_rel
        
        if output_file.exists():
            subjects.append({
                'id': subject_id,
                'input_file': output_file,
                'subject_folder': row['subject_folder'],
                'scan_type': row['scan_type']
            })
        else:
            print(f"  Warning: NIfTI file not found for {subject_id}: {output_file}")
    
    return subjects

def print_progress_summary(tracker_file, config):
    """Print summary of progress"""
    df = pd.read_csv(tracker_file, dtype=str)
    
    total = len(df)
    completed = len(df[df['status'] == 'success'])
    failed = len(df[df['status'] == 'failed'])
    pending = len(df[df['status'] == 'pending'])
    processing = len(df[df['status'] == 'processing'])
    
    print("\n" + "="*60)
    print("MRI PROCESSING SUMMARY")
    print("="*60)
    print(f"Total subjects:       {total}")
    print(f"Completed:            {completed} ({completed/total*100:.1f}%)")
    print(f"Failed:               {failed} ({failed/total*100:.1f}%)")
    print(f"Pending:              {pending} ({pending/total*100:.1f}%)")
    print(f"Currently processing: {processing}")
    print("="*60)
    
    if completed > 0:
        completed_df = df[df['status'] == 'success'].copy()
        completed_df['processing_time_sec'] = pd.to_numeric(completed_df['processing_time_sec'], errors='coerce')
        completed_df = completed_df.dropna(subset=['processing_time_sec'])
        
        if len(completed_df) > 0:
            avg_time = completed_df['processing_time_sec'].mean()
            remaining = pending + processing
            n_workers = config['processing']['n_workers']
            est_time_hours = (remaining * avg_time) / 3600 / n_workers
            print(f"Avg processing time:  {avg_time:.1f} sec/subject")
            print(f"Estimated remaining:  {est_time_hours:.1f} hours ({est_time_hours*60:.0f} min)")
            print("="*60)
    
    if failed > 0:
        print("\nFailed subjects:")
        failed_df = df[df['status'] == 'failed'][['subject_id', 'error_message']]
        for _, row in failed_df.head(5).iterrows():
            error_short = row['error_message'][:60] + '...' if len(row['error_message']) > 60 else row['error_message']
            print(f"  - {row['subject_id']}: {error_short}")
        if failed > 5:
            print(f"  ... and {failed - 5} more")
        print("="*60)

def run_batch_processing(config):
    """Main batch processing function"""
    
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("Setting up SPM environment...")
    process_sMRI.setup_spm_environment(config)
    
    print("\nDiscovering MRI subjects from conversion log...")
    subjects = discover_mri_subjects(config)
    print(f"✓ Found {len(subjects)} MRI subjects to process")
    
    if len(subjects) == 0:
        print("No MRI subjects found. Exiting.")
        return
    
    print("\nInitializing progress tracker...")
    tracker_file = init_progress_tracker(log_dir, subjects)
    
    df = pd.read_csv(tracker_file)
    if config['processing']['skip_existing']:
        pending_ids = df[df['status'].isin(['pending', 'failed'])]['subject_id'].tolist()
        subjects = [s for s in subjects if s['id'] in pending_ids]
        print(f"✓ Skipping completed subjects. {len(subjects)} subjects remaining to process.")
    
    if len(subjects) == 0:
        print("\nAll MRI subjects already processed!")
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
        if n_workers == 1:
            print("  Using sequential processing (n_workers=1)")
            results = [process_subject_wrapper(arg) for arg in args]
        else:
            print(f"  Using parallel processing with {n_workers} workers")
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

class SMRIProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters

    def launch(self):
        return run_batch_processing(self.parameters)

    def __call__(self):
        return self.launch()


if __name__ == '__main__':
    print("="*60)
    print("BATCH MRI PREPROCESSING PIPELINE")
    print("="*60)

    config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'preprocess' / 'processing.yaml'
    config = load_config(str(config_path))
    run_batch_processing(config)
