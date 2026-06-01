import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data import clean_dataset


def test_clean_dataset_keeps_required_columns():
    raw = pd.DataFrame(
        {
            "Size": [1200, 1500],
            "NumberOfRooms": [2, 3],
            "Bathrooms": [1, 2],
            "Floors": [1, 2],
            "ParkingSpaces": [1, 2],
            "PropertyAge": [10, 20],
            "DistanceToCityCenter": [8.5, 2.0],
            "Location": ["Suburb", "Downtown"],
            "Amenities": ["Garden", None],
            "FurnishingStatus": ["Semi-Furnished", "Fully-Furnished"],
            "Price": ["250000", "not-a-price"],
            "UnusedColumn": ["ignored", "ignored"],
        }
    )

    cleaned = clean_dataset(raw)

    assert list(cleaned.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert len(cleaned) == 1
    assert cleaned[TARGET_COLUMN].iloc[0] == 250000


def test_clean_dataset_requires_expected_columns():
    raw = pd.DataFrame({"Size": [1200], "Price": [250000]})

    with pytest.raises(ValueError, match="missing required columns"):
        clean_dataset(raw)
