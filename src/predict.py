from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_MODEL_PATH
from .data import load_prediction_input


def predict(input_path: str | Path, model_path: Path = DEFAULT_MODEL_PATH, output_path: str | Path | None = None) -> pd.DataFrame:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train a model first with: python -m src.train"
        )

    model = joblib.load(model_path)
    input_df = load_prediction_input(input_path)

    result = input_df.copy()
    result["PredictedPrice"] = model.predict(input_df)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict house prices from a CSV file.")
    parser.add_argument("--input", required=True, help="CSV file containing house feature rows.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved model.")
    parser.add_argument("--output", default=None, help="Optional output CSV path for predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict(args.input, args.model_path, args.output)
    print(predictions.to_string(index=False))

    if args.output:
        print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
