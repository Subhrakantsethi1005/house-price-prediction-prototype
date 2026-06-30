# Contributing to House Price Prediction Prototype

Thanks for your interest in contributing! This project is a learning-focused,
end-to-end machine learning prototype for predicting house prices from a
synthetic dataset. Contributions that improve clarity, robustness, or model
quality are welcome.

## Getting Started

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/Subhrakantsethi1005/house-price-prediction-prototype.git
   cd house-price-prediction-prototype
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Generate the synthetic dataset and train the models as described in the
   `README.md`.

## How to Contribute

- **Bugs:** open an issue describing the problem, steps to reproduce, and your
  environment.
- **Features / models:** new regression models, feature-engineering ideas, or
  evaluation metrics are welcome — please open an issue to discuss first.
- **Docs:** improvements to the README, `MODEL_CARD.md`, or `DATA_SCHEMA.md` are
  always appreciated.

## Development Guidelines

- Keep code modular and consistent with the existing structure in `src/`.
- Add or update unit tests in `tests/` for any new behaviour.
- Run the test suite before submitting:
  ```bash
  pytest
  ```
- Ensure the CI workflow (`.github/workflows/`) passes.

## Pull Requests

1. Create a descriptive branch (e.g. `feature/xgboost-model`).
2. Make focused commits with clear messages.
3. Open a pull request summarising the change and why it's useful.

## Note

The synthetic data in this project is for learning and experimentation only and
must not be used for real-world financial decisions.

## Maintainers

- Subhrakant Sethi
- Ayush Singh
