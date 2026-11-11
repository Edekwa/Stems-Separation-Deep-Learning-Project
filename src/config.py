
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DIR = RAW_DATA_DIR / "train"
TEST_DIR = RAW_DATA_DIR / "test"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Audio specifications
SAMPLE_RATE = 44100
N_CHANNELS = 2

STEMS = ("vocals", "drums", "bass", "other")
AUDIO_FILES = ("mixture", *STEMS)
