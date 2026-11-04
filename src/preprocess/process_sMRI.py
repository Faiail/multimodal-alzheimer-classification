import nipype.interfaces.spm as spm
import nipype.interfaces.ants as ants
from pathlib import Path
import os
import traceback
import gzip
import yaml
import subprocess

import logging
logging.getLogger('nipype').setLevel(logging.WARNING)
logging.getLogger('nipype.interface').setLevel(logging.WARNING)

def _patched_version_getter(self):
    return '25.01.02'

try:
    from nipype.interfaces.spm.base import SPMCommand
    SPMCommand.version = property(_patched_version_getter)
except Exception as e:
    print(f"Warning: Could not patch SPM version getter: {e}")

def load_config(config_path):
    """Load configuration from YAML file and resolve relative paths to absolute"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    workspace_root = Path(config_path).resolve().parent.parent
    
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str) and not os.path.isabs(value):
                config['paths'][key] = str(workspace_root / value)
    
    return config

def setup_spm_environment(config):
    """Configure SPM environment variables"""
    full_spm_command = f"{config['spm']['script_path']} {config['spm']['mcr_path']}"
    os.environ['SPMMCRCMD'] = full_spm_command
    os.environ['FORCE_SPMMCR'] = '1'
    
    os.environ['NIPYPE_NO_ET'] = '1'
    
    print(f"  SPM command configured: {full_spm_command}")
    
    try:
        _ = spm.NewSegment()
        print("  ✓ SPM interface initialized successfully")
    except Exception as e:
        print(f"  ⚠ SPM version check failed (expected with SPM25): {str(e)[:100]}...")
        print("  Continuing with SPM processing (version check disabled)")

def process_single_subject(input_file, subject_id, config, skip_existing=True):
    """
    Process a single subject through N4 correction and segmentation.
    
    Args:
        input_file: Path to input NIfTI file
        subject_id: Subject identifier
        config: Configuration dictionary
        skip_existing: Skip if outputs already exist
    
    Returns:
        dict with status and output paths
    """
    
    n4_dir = Path(config['paths']['processed']) / 'n4'
    seg_dir = Path(config['paths']['processed']) / 'segmentation' / subject_id
    
    n4_output = n4_dir / f'{subject_id}_n4.nii.gz'
    n4_nii = n4_dir / f'{subject_id}_n4.nii'
    
    n4_nii_in_seg = seg_dir / f'{subject_id}_n4.nii'
    
    try:
        n4_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    
    try:
        seg_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    
    result = {
        'subject_id': subject_id,
        'status': 'pending',
        'n4_file': None,
        'segmentation_files': {},
        'error': None
    }
    
    try:
        if skip_existing and n4_output.exists():
            print(f"  N4 output exists, skipping N4 step")
            result['n4_file'] = n4_output
        else:
            print(f"  Running N4 Bias Correction...")
            n4 = ants.N4BiasFieldCorrection()
            n4.inputs.dimension = 3
            n4.inputs.input_image = str(input_file)
            n4.inputs.output_image = str(n4_output)
            n4.inputs.bspline_fitting_distance = config['n4']['bspline_fitting_distance']
            n4.inputs.shrink_factor = config['n4']['shrink_factor']
            n4.inputs.n_iterations = config['n4']['n_iterations']
            
            try:
                n4.run()
            except FileNotFoundError as e:
                if not n4_output.exists():
                    raise
                print(f"  ⚠ Nipype validation failed but file exists (path with spaces issue)")
            
            result['n4_file'] = n4_output
            print(f"  ✓ N4 completed")
        
        if not n4_nii.exists():
            print(f"  Unzipping N4 output...")
            with gzip.open(str(n4_output), 'rb') as f_in:
                with open(str(n4_nii), 'wb') as f_out:
                    f_out.write(f_in.read())
        
        expected_files = list(seg_dir.glob('wc1*.nii'))
        if skip_existing and len(expected_files) > 0:
            print(f"  Segmentation outputs exist, skipping segmentation")
            result['segmentation_files'] = {'exists': True}
        else:
            print(f"  Copying N4 file to segmentation directory...")
            import shutil
            shutil.copy2(str(n4_nii), str(n4_nii_in_seg))
            
            if not n4_nii_in_seg.exists():
                raise FileNotFoundError(f"Failed to copy N4 file to {n4_nii_in_seg}")
            
            n4_absolute_path = str(n4_nii_in_seg.absolute())
            
            print(f"  Running SPM NewSegment...")
            
            original_cwd = os.getcwd()
            os.chdir(str(seg_dir))
            
            try:
                import subprocess
                
                matlab_script = seg_dir / 'batch_newsegment.m'
                tpm_file = config['spm']['tpm_file']
                
                if not Path(tpm_file).exists():
                    raise FileNotFoundError(f"TPM file not found: {tpm_file}")
                
                with open(matlab_script, 'w') as f:
                    f.write("fprintf(1,'Executing batch_newsegment.m at %s:\\n',datestr(now));\n")
                    f.write("ver,\n")
                    f.write("try,\n")
                    f.write("    if isempty(which('spm')),\n")
                    f.write("         throw(MException('SPMCheck:NotFound', 'SPM not in matlab path'));\n")
                    f.write("    end\n")
                    f.write("    [name, version] = spm('ver');\n")
                    f.write("    fprintf('SPM version: %s Release: %s\\n',name, version);\n")
                    f.write("    spm('Defaults','fMRI');\n")
                    f.write("    spm_jobman('initcfg');\n")
                    f.write("    spm_get_defaults('cmdline', 1);\n")
                    f.write(f"    jobs{{1}}.spm.tools.preproc8.channel(1).vols = {{'{n4_absolute_path},1'}};\n")
                    for i in range(1, 7):
                        ngaus = [2, 2, 2, 3, 4, 2][i-1]
                        native = ['1', '1', '1', '0', '0', '0'][i-1]
                        warped = ['1', '1', '1', '0', '0', '0'][i-1]
                        f.write(f"    jobs{{1}}.spm.tools.preproc8.tissue({i}).tpm = {{'{tpm_file},{i}'}};\n")
                        f.write(f"    jobs{{1}}.spm.tools.preproc8.tissue({i}).ngaus = {ngaus};\n")
                        f.write(f"    jobs{{1}}.spm.tools.preproc8.tissue({i}).native(1) = {native};\n")
                        f.write(f"    jobs{{1}}.spm.tools.preproc8.tissue({i}).native(2) = 0;\n")
                        f.write(f"    jobs{{1}}.spm.tools.preproc8.tissue({i}).warped(1) = {warped};\n")
                        f.write(f"    jobs{{1}}.spm.tools.preproc8.tissue({i}).warped(2) = 0;\n")
                    f.write("    jobs{1}.spm.tools.preproc8.warp.write(1) = 0;\n")
                    f.write("    jobs{1}.spm.tools.preproc8.warp.write(2) = 1;\n")
                    f.write("    spm_jobman('run', jobs);\n")
                    f.write("    fprintf('SUCCESS\\n');\n")
                    f.write(",catch ME,\n")
                    f.write("    fprintf(2,'MATLAB code threw an exception:\\n');\n")
                    f.write("    fprintf(2,'%s\\n',ME.message);\n")
                    f.write("    if length(ME.stack) ~= 0, fprintf(2,'File:%s\\nName:%s\\nLine:%d\\n',ME.stack.file,ME.stack.name,ME.stack.line);, end;\n")
                    f.write("end\n")
                
                print(f"  Executing SPM segmentation...")
                spm_cmd = [
                    config['spm']['script_path'],
                    config['spm']['mcr_path'],
                    'batch',
                    str(matlab_script)
                ]
                
                spm_result = subprocess.run(
                    spm_cmd,
                    capture_output=True,
                    text=True,
                    timeout=config['processing'].get('spm_timeout', 1800)
                )
                
                print(f"  SPM return code: {spm_result.returncode}")
                if spm_result.returncode != 0:
                    print(f"  SPM stderr: {spm_result.stderr[:500]}")
                
                if 'SUCCESS' in spm_result.stdout or spm_result.returncode == 0:
                    print(f"  ✓ SPM execution completed")
                else:
                    print(f"  ⚠ SPM may have failed, checking outputs...")
                
                print(f"  Checking for output files in {seg_dir}...")
                all_files = list(seg_dir.glob('*.nii'))
                print(f"  Total .nii files in directory: {len(all_files)}")
                
                if all_files:
                    print(f"  Files found: {[f.name for f in all_files[:10]]}")
                
                created_files = {
                    'wc1': list(seg_dir.glob(f'wc1*.nii')),
                    'wc2': list(seg_dir.glob(f'wc2*.nii')),
                    'wc3': list(seg_dir.glob(f'wc3*.nii')),
                    'c1': list(seg_dir.glob(f'c1*.nii')),
                    'c2': list(seg_dir.glob(f'c2*.nii')),
                    'c3': list(seg_dir.glob(f'c3*.nii')),
                    'y': list(seg_dir.glob(f'y_*.nii'))
                }
                
                print(f"  Segmentation outputs found:")
                print(f"    wc1: {len(created_files['wc1'])}")
                print(f"    wc2: {len(created_files['wc2'])}")
                print(f"    wc3: {len(created_files['wc3'])}")
                
                if created_files['wc1'] and created_files['wc2'] and created_files['wc3']:
                    result['segmentation_files'] = created_files
                    print(f"  ✓ Segmentation completed (outputs in {seg_dir.name}/)")
                else:
                    mat_files = list(seg_dir.glob('*.mat'))
                    print(f"  MAT files found: {len(mat_files)}")
                    if mat_files:
                        print(f"  MAT files: {[f.name for f in mat_files]}")
                    
                    raise FileNotFoundError(
                        f"SPM completed but expected output files not found in {seg_dir}. "
                        f"Found wc1: {len(created_files['wc1'])}, wc2: {len(created_files['wc2'])}, wc3: {len(created_files['wc3'])}. "
                        f"Total .nii files: {len(all_files)}"
                    )
                
            finally:
                os.chdir(original_cwd)
            
            if n4_nii_in_seg.exists():
                os.remove(n4_nii_in_seg)
                print(f"  Cleaned up N4 copy from segmentation directory")
            
            pyscript_files = list(seg_dir.glob('pyscript*.m'))
            for script_file in pyscript_files:
                os.remove(script_file)
            if pyscript_files:
                print(f"  Cleaned up {len(pyscript_files)} temporary MATLAB scripts")
        
        if config.get('cleanup', {}).get('delete_intermediate_n4_unzipped', False):
            if n4_nii.exists():
                os.remove(n4_nii)
                print(f"  Cleaned up N4 unzipped file from n4 directory")
        
        result['status'] = 'success'
        return result
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"  ✗ ERROR: {e}")
        traceback.print_exc()
        return result

if __name__ == '__main__':
    print("--- Starting Single Subject sMRI Preprocessing ---")
    
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'processing_config.yaml'
    config = load_config(config_path)
    
    setup_spm_environment(config)
    
    input_file = Path(config['paths']['processed']) / 'nifti/I274154_MPRAGE_SENSE_20111222093538_301.nii.gz'
    subject_id = 'I274154_MPRAGE_SENSE_20111222093538_301'
    
    result = process_single_subject(input_file, subject_id, config)
    
    if result['status'] == 'success':
        print("\n--- Processing Finished Successfully! ---")
    else:
        print(f"\n--- Processing Failed: {result['error']} ---")

