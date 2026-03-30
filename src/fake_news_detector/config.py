from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

DEFAULT_DATASET_PATH = Path(
    os.getenv("FAKE_NEWS_DATASET", DATA_DIR / "WELFake_Dataset.csv")
)
DEFAULT_ARTIFACTS_DIR = Path(
    os.getenv("FAKE_NEWS_ARTIFACTS", ARTIFACTS_DIR)
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 20_000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
NGRAM_RANGE = (1, 2)

DEFAULT_MODEL_NAME = "LogisticRegression"
