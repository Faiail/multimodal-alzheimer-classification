#!/usr/bin/env python3
"""
Corrects data/processed/all_features_multimodal.csv by integrating longitudinal
information from data/raw/ADNIMERGE_31Dec2025.csv.

Key Corrections:
1. Dynamic Visit-Level Labeling:
   - Resolves contradictory annotations by mapping each scan to its exact visit diagnosis (DX).
   - Corrects scans of MCI patients who converted to Alzheimer's Disease (MCI -> AD)
     from their baseline label 'MCI' to the true post-conversion label 'AD'.
2. Visit-Level Clinical Measurements:
   - Updates AGE and MMSE to match the actual visit date instead of static baseline values.
3. Scanner & Protocol Metadata:
   - Adds COLPROT (ADNI1, ADNI2, ADNI3, ADNIGO) and field_strength (1.5T vs 3.0T)
     for ComBat harmonization and protocol-aware modeling.
4. Preserves backward compatibility:
   - Keeps original_class and label columns matching downstream split and processor schemas.
   - Keeps original_class_static for reference and auditability.
"""

import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = PROJECT_ROOT / "data/processed/all_features_multimodal.csv"
RAW_DATASET_CSV = PROJECT_ROOT / "data/processed/multimodal_raw_dataset.csv"
ADNIMERGE_CSV = PROJECT_ROOT / "data/raw/ADNIMERGE_31Dec2025.csv"


def standardize_diagnosis(dx_str: str) -> str:
    """Standardizes ADNI DX string into canonical 'CN', 'MCI', or 'AD'."""
    if not isinstance(dx_str, str) or pd.isna(dx_str):
        return "CN"
    s = dx_str.strip().upper()
    if s in ["DEMENTIA", "AD", "ALZHEIMER"]:
        return "AD"
    elif s in ["MCI", "LMCI", "EMCI"]:
        return "MCI"
    elif s in ["CN", "NL", "NORMAL", "SMC"]:
        return "CN"
    return "CN"


def main():
    print("=" * 75)
    print(" DATASET CORRECTION: all_features_multimodal.csv & multimodal_raw_dataset.csv")
    print("=" * 75)
    print(f"• Input Dataset:   {INPUT_CSV}")
    print(f"• Clinical Source: {ADNIMERGE_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    if not ADNIMERGE_CSV.exists():
        raise FileNotFoundError(f"ADNIMERGE file not found: {ADNIMERGE_CSV}")

    # 1. Back up original files
    bak_input = INPUT_CSV.with_suffix(".csv.bak")
    if not bak_input.exists():
        shutil.copy2(INPUT_CSV, bak_input)
        print(f"✓ Backup created: {bak_input}")
    else:
        print(f"✓ Backup already exists: {bak_input}")

    if RAW_DATASET_CSV.exists():
        bak_raw = RAW_DATASET_CSV.with_suffix(".csv.bak")
        if not bak_raw.exists():
            shutil.copy2(RAW_DATASET_CSV, bak_raw)
            print(f"✓ Backup created: {bak_raw}")

    # 2. Load Datasets
    print("\n[1/4] Loading CSV data...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    adni = pd.read_csv(ADNIMERGE_CSV, low_memory=False)
    adni["EXAMDATE"] = pd.to_datetime(adni["EXAMDATE"], errors="coerce")

    print(f"  • Total scans in dataset: {len(df)} ({df['PTID'].nunique()} unique patients)")
    print(f"  • Total ADNIMERGE records: {len(adni)}")

    # Filter ADNIMERGE for our patient cohort
    cohort_ptids = set(df["PTID"])
    adni_cohort = adni[adni["PTID"].isin(cohort_ptids)].sort_values(["PTID", "EXAMDATE"])

    # 3. Process Patients & Correct Annotations
    print("\n[2/4] Aligning scans with longitudinal visits and resolving label conflicts...")

    # Preserve historical static class
    if "original_class_static" not in df.columns:
        df["original_class_static"] = df["original_class"]

    # Prep new columns
    corrected_classes = []
    corrected_labels = []
    updated_ages = []
    updated_mmses = []
    colprots = []
    field_strengths = []
    viscodes = []
    examdates = []
    is_converter_list = []
    conv_status_list = []

    # Counters
    scans_reclassified = 0
    mci_to_ad_count = 0
    cn_to_mci_count = 0
    cn_to_ad_count = 0
    converter_patients_found = 0

    df["img_num"] = df["scan_id"].astype(str).str.replace("I", "", regex=False).astype(int)

    for ptid, group in df.groupby("PTID", sort=False):
        group_sorted = group.sort_values("img_num")
        n_scans = len(group_sorted)

        # Cluster scans into sessions (scans with delta img_num <= 500 belong to the same session)
        diffs = group_sorted["img_num"].diff().fillna(1000)
        session_ids = (diffs > 500).cumsum()
        unique_sessions = session_ids.unique()
        n_sessions = len(unique_sessions)

        # Get patient records in ADNIMERGE
        pt_adni = adni_cohort[adni_cohort["PTID"] == ptid].dropna(subset=["DX"]).copy()
        
        # Check conversion trajectory
        is_converter = False
        conv_date = None
        if len(pt_adni) > 1:
            bl_dx = standardize_diagnosis(pt_adni["DX"].iloc[0])
            for _, v_row in pt_adni.iterrows():
                cur_dx = standardize_diagnosis(v_row["DX"])
                if bl_dx == "MCI" and cur_dx == "AD":
                    is_converter = True
                    conv_date = v_row["EXAMDATE"]
                    break
                elif bl_dx == "CN" and cur_dx in ["MCI", "AD"]:
                    is_converter = True
                    conv_date = v_row["EXAMDATE"]
                    break

        if is_converter:
            converter_patients_found += 1

        # Prefer visits with valid MRI scans (IMAGEUID not null) if sufficient
        pt_mri_visits = pt_adni.dropna(subset=["IMAGEUID"])
        target_visits = pt_mri_visits if len(pt_mri_visits) >= n_sessions else pt_adni

        # Map each session index to the corresponding visit row
        session_to_visit = {}
        target_list = list(target_visits.iterrows()) if len(target_visits) > 0 else list(pt_adni.iterrows())

        for s_idx, s_id in enumerate(unique_sessions):
            if target_list:
                # Clamp to available visits
                v_idx = min(s_idx, len(target_list) - 1)
                session_to_visit[s_id] = target_list[v_idx][1]
            else:
                session_to_visit[s_id] = None

        # Process each scan in original row order of the group
        for _, scan_row in group.iterrows():
            s_id = session_ids.loc[scan_row.name]
            matched_visit = session_to_visit.get(s_id)
            orig_static = str(scan_row["original_class_static"]).strip().upper()

            if matched_visit is not None:
                dx_raw = matched_visit.get("DX", orig_static)
                dyn_dx = standardize_diagnosis(dx_raw)
                
                # Check visit date
                v_date = matched_visit.get("EXAMDATE", None)
                v_code = str(matched_visit.get("VISCODE", "bl"))
                prot = str(matched_visit.get("COLPROT", "ADNI2"))
                v_age = float(matched_visit.get("AGE", scan_row["AGE"]))
                v_mmse = matched_visit.get("MMSE", scan_row["MMSE"])
                v_mmse = float(v_mmse) if pd.notnull(v_mmse) else float(scan_row["MMSE"])
            else:
                dyn_dx = orig_static
                v_date = None
                v_code = "bl"
                prot = "ADNI2"
                v_age = float(scan_row["AGE"])
                v_mmse = float(scan_row["MMSE"])

            # Map field strength from protocol
            f_strength = "1.5T" if "ADNI1" in prot.upper() else "3.0T"

            # Check if this scan changed diagnosis from static baseline
            if dyn_dx != orig_static:
                scans_reclassified += 1
                if orig_static == "MCI" and dyn_dx == "AD":
                    mci_to_ad_count += 1
                elif orig_static == "CN" and dyn_dx == "MCI":
                    cn_to_mci_count += 1
                elif orig_static == "CN" and dyn_dx == "AD":
                    cn_to_ad_count += 1

            # Determine conversion state
            if is_converter:
                if conv_date is not None and v_date is not None and pd.notnull(v_date):
                    conv_status = "Post-Conversion" if v_date >= conv_date else "Pre-Conversion"
                else:
                    conv_status = "Post-Conversion" if dyn_dx in ["AD", "MCI"] and orig_static != dyn_dx else "Pre-Conversion"
            else:
                conv_status = "Stable"

            # Assign label matching dynamic diagnosis
            label_map = {"CN": 0, "MCI": 1, "AD": 2}
            dyn_label = label_map.get(dyn_dx, 0)

            corrected_classes.append(dyn_dx)
            corrected_labels.append(dyn_label)
            updated_ages.append(round(v_age, 1))
            updated_mmses.append(round(v_mmse, 1))
            colprots.append(prot)
            field_strengths.append(f_strength)
            viscodes.append(v_code)
            examdates.append(str(v_date.date()) if pd.notnull(v_date) else "")
            is_converter_list.append(is_converter)
            conv_status_list.append(conv_status)

    # 4. Update DataFrame
    print("\n[3/4] Updating dataset columns...")
    df["original_class"] = corrected_classes
    df["label"] = corrected_labels
    df["AGE"] = updated_ages
    df["MMSE"] = updated_mmses
    df["COLPROT"] = colprots
    df["field_strength"] = field_strengths
    df["VISCODE"] = viscodes
    df["EXAMDATE"] = examdates
    df["is_converter"] = is_converter_list
    df["conversion_status"] = conv_status_list

    # Drop temporary column
    if "img_num" in df.columns:
        df = df.drop(columns=["img_num"])

    # 5. Save Corrected CSVs
    print("\n[4/4] Saving corrected files...")
    df.to_csv(INPUT_CSV, index=False)
    print(f"✓ Saved updated dataset: {INPUT_CSV}")

    if RAW_DATASET_CSV.exists():
        df.to_csv(RAW_DATASET_CSV, index=False)
        print(f"✓ Synced: {RAW_DATASET_CSV}")

    # Summary Statistics
    print("\n" + "=" * 75)
    print(" SUMMARY OF CORRECTIONS")
    print("=" * 75)
    print(f"• Total scans processed:              {len(df)}")
    print(f"• Converter patients detected:        {converter_patients_found} / {df['PTID'].nunique()} ({converter_patients_found/df['PTID'].nunique()*100:.1f}%)")
    print(f"• Total scans with corrected label:   {scans_reclassified} ({scans_reclassified/len(df)*100:.1f}%)")
    print(f"  - MCI scans corrected to AD:        {mci_to_ad_count} (Progressed patients with true Alzheimer's)")
    print(f"  - CN scans corrected to MCI:        {cn_to_mci_count} (Progressed patients with cognitive decline)")
    print(f"  - CN scans corrected to AD:         {cn_to_ad_count} (Direct conversion to Alzheimer's)")
    
    print("\n• New Class Distribution (original_class):")
    for cls, cnt in df["original_class"].value_counts().items():
        print(f"  - {cls:<5}: {cnt:>5} scans ({cnt/len(df)*100:>5.1f}%) [Old static: {df['original_class_static'].value_counts().get(cls, 0)}]")

    print("\n• Field Strength Distribution (field_strength):")
    for fs, cnt in df["field_strength"].value_counts().items():
        print(f"  - {fs:<5}: {cnt:>5} scans ({cnt/len(df)*100:>5.1f}%)")

    print("\n• Protocol Phase Distribution (COLPROT):")
    for prot, cnt in df["COLPROT"].value_counts().items():
        print(f"  - {prot:<8}: {cnt:>5} scans ({cnt/len(df)*100:>5.1f}%)")

    print("=" * 75)
    print("CORRECTION COMPLETED SUCCESSFULLY!")
    print("Next steps:")
    print("  1. Regenerate task splits: ./scripts/preprocess/splits.sh")
    print("  2. Re-run experiments:     CUDA_VISIBLE_DEVICES=0 ./scripts/experiment/lassonet_local.sh (or other models)")
    print("=" * 75)


if __name__ == "__main__":
    main()
