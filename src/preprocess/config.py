from pathlib import Path
from src.utils.general import DEFAULT_DATA_ROOT, resolve_data_path

DATA_ROOT = Path(resolve_data_path(DEFAULT_DATA_ROOT))
RAW_DATA_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
OUTPUT_DIR = Path("data/processed")
FINAL_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

FILE_ADNIMERGE = RAW_DATA_DIR / "ADNIMERGE_31Dec2025.csv"
FILE_UPENN = RAW_DATA_DIR / "UPENNBIOMK_ROCHE_ELECSYS_31Dec2025.csv"
FILE_FEATURES = FINAL_DIR / "all_features.csv"
