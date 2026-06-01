# Project Report

## Objective

The goal of this project is to build a clean, reproducible machine learning prototype for house price prediction. The dataset is synthetic, but the project is structured like a professional ML repository so another user can generate data, train a model, evaluate it, and run predictions locally.

## Why Synthetic Data

The original dataset was generated for learning purposes. Instead of hiding that, this project makes the synthetic nature explicit and improves the generator so the model learns realistic relationships.

This keeps the project honest while still demonstrating practical ML engineering skills.

## Methodology

1. Generate a synthetic housing dataset with realistic pricing logic.
2. Validate the required schema.
3. Split the data into train and test sets.
4. Preprocess numeric and categorical features with a scikit-learn pipeline.
5. Train several regression models.
6. Select the best model using MAE.
7. Save the trained model and metrics.
8. Provide a prediction CLI for new CSV input.

## Skills Demonstrated

- Python project organization
- Synthetic data generation
- Regression modeling
- Data preprocessing with `Pipeline` and `ColumnTransformer`
- Model evaluation
- Model serialization with `joblib`
- CLI-based training and prediction
- Unit testing
- GitHub Actions CI
- Documentation and model limitations

## Professional Positioning

This should be presented as a prototype, not as a deployed real-estate pricing product. The strongest part of the project is the end-to-end ML workflow and the transparency around assumptions and limitations.

