from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from .synthetic_data import generate_synthetic_dataset


def load_dataset(data_path: str | Path | None = None, generate_if_missing: bool = True) -> pd.DataFrame:
    """Load a local dataset, or generate the default synthetic dataset when needed."""
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    if not path.exists():
        if data_path or not generate_if_missing:
            raise FileNotFoundError(f"Dataset not found: {path}")
        generate_synthetic_dataset(path)

    df = pd.read_csv(path)
    return clean_dataset(df)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and keep the rows usable for model training."""
    missing_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df.copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def load_prediction_input(input_path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV file used for prediction."""
    df = pd.read_csv(input_path)
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Prediction input is missing required columns: {missing_columns}")

    return df[FEATURE_COLUMNS]
