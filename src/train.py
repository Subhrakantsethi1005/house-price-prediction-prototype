from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from .data import load_dataset


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
        ),
    }


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def train(data_path: str | Path | None = None, model_path: Path = DEFAULT_MODEL_PATH, metrics_path: Path = DEFAULT_METRICS_PATH) -> dict:
    df = load_dataset(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    results = {}
    best_name = None
    best_pipeline = None
    best_mae = np.inf

    for name, model in candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_regression(y_test, predictions)
        results[name] = metrics

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name
            best_pipeline = pipeline

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path)

    report = {
        "best_model": best_name,
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "metrics": results,
        "model_path": str(model_path),
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the house price prediction model.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional path to a local training CSV.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the trained model.")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH, help="Where to save model metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train(args.data_path, args.model_path, args.metrics_path)
    print(f"Best model: {report['best_model']}")
    print(f"Saved model to: {report['model_path']}")
    print(f"Saved metrics to: {args.metrics_path}")

    for model_name, metrics in report["metrics"].items():
        print(
            f"{model_name}: "
            f"MAE={metrics['mae']:.2f}, "
            f"RMSE={metrics['rmse']:.2f}, "
            f"R2={metrics['r2']:.4f}, "
            f"MAPE={metrics['mape']:.4f}"
        )


if __name__ == "__main__":
    main()
