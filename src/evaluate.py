from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .config import DEFAULT_MODEL_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from .data import load_dataset
from .train import evaluate_regression


def evaluate(data_path: str | Path | None = None, model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, float]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train a model first with: python -m src.train"
        )

    df = load_dataset(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = joblib.load(model_path)
    predictions = model.predict(X)
    return evaluate_regression(y, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved house price prediction model.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional path to a local evaluation CSV.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.data_path, args.model_path)
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.4f}")


if __name__ == "__main__":
    main()
