from pathlib import Path

from dotenv import load_dotenv

WORK_DIR = Path(__file__).parents[2]

DATA_DIR = WORK_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RESULT_DIR = WORK_DIR / "results"
FIGURE_RESULT_DIR = RESULT_DIR / "figures"
MODEL_RESULT_DIR = RESULT_DIR / "models"
TABLE_RESULT_DIR = RESULT_DIR / "tables"

load_dotenv(dotenv_path=WORK_DIR / ".env")
