from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_DATA_PATH = DATA_DIR / "house_data.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "house_price_model.joblib"
DEFAULT_METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET_COLUMN = "Price"

NUMERIC_FEATURES = [
    "Size",
    "NumberOfRooms",
    "Bathrooms",
    "Floors",
    "ParkingSpaces",
    "PropertyAge",
    "DistanceToCityCenter",
]
CATEGORICAL_FEATURES = ["Location", "Amenities", "FurnishingStatus"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
DEFAULT_SYNTHETIC_ROWS = 5000
