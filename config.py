from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "main"
ASSET_NAMES = ["AAPL", "AMZN", "JPM"]
INDEX_NAME = "EW_INDEX"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

HORIZON = 1
HORIZONS = [1, 5, 22]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
MODEL_SET = "MALL"  # "MHAR" or "MALL"
NN_SEEDS = 5         # quick test: 5; final run: 20 or 100
RANDOM_STATE = 42
