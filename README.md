# House Price Prediction Prototype

This project is a machine learning prototype for predicting house prices from a synthetic dataset. It is designed for learning, experimentation, and research-style demonstrations of an end-to-end regression workflow.

Important: the dataset is synthetic. Predictions from this project should not be used for real-world property valuation, financial decisions, or investment decisions.

See [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md) for details about how the generated prices are constructed.

Additional documentation:

- [DATA_SCHEMA.md](DATA_SCHEMA.md)
- [MODEL_CARD.md](MODEL_CARD.md)
- [PROJECT_REPORT.md](PROJECT_REPORT.md)

## What This Project Does

The project generates realistic synthetic housing data and trains regression models to predict `Price`.

The synthetic data intentionally includes relationships such as:

- Bigger houses generally cost more.
- Downtown and waterfront homes receive location premiums.
- Older homes usually lose value, but premium locations reduce that penalty.
- Amenities such as gardens, gyms, pools, and smart-home features add value.
- Furnishing status affects price.
- Homes farther from the city center usually cost less.
- Random noise is added so the model does not learn a perfect formula.

## Features Used

The model uses these input columns:

```text
Size
NumberOfRooms
Bathrooms
Floors
ParkingSpaces
PropertyAge
DistanceToCityCenter
Location
Amenities
FurnishingStatus
```

The target column is:

```text
Price
```

## Project Structure

```text
.
|-- README.md
|-- SYNTHETIC_DATA.md
|-- DATA_SCHEMA.md
|-- MODEL_CARD.md
|-- PROJECT_REPORT.md
|-- CONTRIBUTING.md
|-- AUTHORS.md
|-- requirements.txt
|-- pytest.ini
|-- .gitignore
|-- LICENSE
|-- data/
|-- examples/
|   |-- sample_houses.csv
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- synthetic_data.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|-- tests/
|   |-- test_data.py
|-- .github/
|   |-- workflows/
|       |-- python-ci.yml
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate Synthetic Data

Generate the default dataset:

```bash
python -m src.synthetic_data
```

Generate a larger dataset:

```bash
python -m src.synthetic_data --rows 20000 --output data/house_data.csv
```

The generated CSV is saved to `data/house_data.csv`. Data files are ignored by Git so generated artifacts do not get uploaded accidentally.

## Train The Model

```bash
python -m src.train
```

If `data/house_data.csv` does not exist, training will generate it automatically.

Train using another CSV:

```bash
python -m src.train --data-path data/my_house_data.csv
```

The training script compares:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- GradientBoostingRegressor

The best model by MAE is saved to:

```text
models/house_price_model.joblib
```

Metrics are saved to:

```text
reports/metrics.json
```

## Expected Output

After training, the terminal prints the best model and metrics for each candidate model. Because the data is synthetic and reproducible, results should be stable across runs with the same package versions.

Example:

```text
Best model: gradient_boosting
linear_regression: MAE=56028.62, RMSE=72629.74, R2=0.9310, MAPE=0.1278
ridge: MAE=56019.73, RMSE=72626.73, R2=0.9310, MAPE=0.1277
random_forest: MAE=52637.68, RMSE=71051.87, R2=0.9340, MAPE=0.1017
gradient_boosting: MAE=46454.13, RMSE=63572.79, R2=0.9471, MAPE=0.0879
```

These metrics are from a reproducible synthetic dataset generated with the default random seed. They show that the model learns the simulated pricing relationships, not that it predicts real market prices.

## Evaluate The Saved Model

```bash
python -m src.evaluate
```

## Make Predictions

Predict from the sample input file:

```bash
python -m src.predict --input examples/sample_houses.csv
```

Save predictions to a CSV:

```bash
python -m src.predict --input examples/sample_houses.csv --output reports/predictions.csv
```

## Run Tests

```bash
pytest
```

## Future Improvements

Useful next additions:

- Hyperparameter tuning with `RandomizedSearchCV`.
- Cross-validation instead of a single train/test split.
- Feature importance reports.
- Prediction confidence intervals.
- A Streamlit app for interactive predictions.
- A notebook comparing synthetic and real-world housing datasets.
- Model versioning with date-stamped artifacts.
