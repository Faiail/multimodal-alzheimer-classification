from pathlib import Path

RAW_DATA_DIR = Path("data/01_raw")
OUTPUT_DIR = Path("data/02_processed")
FINAL_DIR = Path("data/04_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

FILE_ADNIMERGE = RAW_DATA_DIR / "ADNIMERGE_31Dec2025.csv"
FILE_UPENN = RAW_DATA_DIR / "UPENNBIOMK_ROCHE_ELECSYS_31Dec2025.csv"
FILE_FEATURES = FINAL_DIR / "all_features.csv"