#!/usr/bin/env python3
"""
Applies neuroCombat harmonization to Multi-Atlas feature representations
to remove magnetic field strength (1.5T vs 3.0T) and scanner batch effects
while preserving biological covariates (Diagnosis, Age, Gender).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from neuroCombat import neuroCombat
except ImportError:
    print("Error: neuroCombat not found. Install it with: uv pip install neuroCombat")
    sys.exit(1)

INPUT_CSV = PROJECT_ROOT / "data/processed/all_features_multimodal.csv"
MULTIATLAS_CSV = PROJECT_ROOT / "data/processed/all_features_multiatlas.csv"
OUTPUT_HARMONIZED_CSV = PROJECT_ROOT / "data/processed/all_features_multiatlas_harmonized.csv"


def main():
    print("=" * 75)
    print(" NEUROCOMBAT HARMONIZATION (1.5T vs 3.0T)")
    print("=" * 75)
    print(f"• Multimodal Dataset: {INPUT_CSV}")
    print(f"• Multi-Atlas Source: {MULTIATLAS_CSV}")
    print(f"• Output Harmonized:  {OUTPUT_HARMONIZED_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}")
    if not MULTIATLAS_CSV.exists():
        print(f"Note: {MULTIATLAS_CSV} not yet generated. Run multi_atlas_processor first if needed.")
        return

    meta_df = pd.read_csv(INPUT_CSV, low_memory=False)
    feat_df = pd.read_csv(MULTIATLAS_CSV, low_memory=False)

    print(f"  • Scans with metadata: {len(meta_df)}")
    print(f"  • Scans with features: {len(feat_df)}")

    # Merge on scan_id or safetensor_file
    merge_col = "scan_id" if "scan_id" in feat_df.columns else ("safetensor_file" if "safetensor_file" in feat_df.columns else None)
    if merge_col is None:
        print("Error: Cannot find merge column in feature CSV.")
        return

    merged = pd.merge(feat_df, meta_df[[merge_col, "field_strength", "original_class", "AGE", "PTGENDER"]], on=merge_col, how="inner")
    print(f"  • Merged cohort size: {len(merged)}")

    # Identify numerical feature columns
    feature_cols = [c for c in feat_df.columns if c not in ["scan_id", "PTID", "RID", "safetensor_file", "label", "original_class"]]
    print(f"  • Features to harmonize: {len(feature_cols)} ROIs")

    # Transpose for neuroCombat: shape (features, samples)
    data = merged[feature_cols].values.T.astype(np.float32)

    # Covariates DataFrame
    covars = pd.DataFrame({
        "batch": merged["field_strength"].astype(str),
        "diagnosis": merged["original_class"].astype(str),
        "gender": merged["PTGENDER"].astype(str),
        "age": merged["AGE"].astype(float)
    })

    print(f"\nBatch distribution:")
    print(covars["batch"].value_counts())

    print("\nRunning neuroCombat (adjusting batch variance, preserving diagnosis, age, gender)...")
    res = neuroCombat(
        dat=data,
        covars=covars,
        batch_col="batch",
        categorical_cols=["diagnosis", "gender"],
        continuous_cols=["age"]
    )

    harmonized_data = res["data"].T  # Back to (samples, features)
    
    # Construct output DataFrame
    out_df = merged.copy()
    out_df[feature_cols] = harmonized_data
    out_df.to_csv(OUTPUT_HARMONIZED_CSV, index=False)

    print(f"✓ Harmonization complete! Output saved to: {OUTPUT_HARMONIZED_CSV}")
    print("=" * 75)


if __name__ == "__main__":
    main()
