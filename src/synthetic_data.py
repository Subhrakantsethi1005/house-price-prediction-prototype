from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from .config import DEFAULT_DATA_PATH, DEFAULT_SYNTHETIC_ROWS, FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN


LOCATIONS = {
    "Rural": 0.82,
    "Suburb": 1.00,
    "Downtown": 1.28,
    "Waterfront": 1.45,
}

AMENITIES = {
    "None": 0,
    "Garden": 18000,
    "Gym": 22000,
    "Pool": 35000,
    "SmartHome": 42000,
}

FURNISHING_STATUS = {
    "Unfurnished": 0,
    "Semi-Furnished": 28000,
    "Fully-Furnished": 55000,
}


def clipped(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def choose_location(rng: random.Random) -> str:
    return rng.choices(
        population=["Rural", "Suburb", "Downtown", "Waterfront"],
        weights=[0.25, 0.38, 0.27, 0.10],
        k=1,
    )[0]


def choose_amenity(rng: random.Random) -> str:
    return rng.choices(
        population=["None", "Garden", "Gym", "Pool", "SmartHome"],
        weights=[0.25, 0.25, 0.18, 0.20, 0.12],
        k=1,
    )[0]


def choose_furnishing_status(rng: random.Random) -> str:
    return rng.choices(
        population=["Unfurnished", "Semi-Furnished", "Fully-Furnished"],
        weights=[0.34, 0.43, 0.23],
        k=1,
    )[0]


def generate_row(rng: random.Random) -> dict[str, object]:
    location = choose_location(rng)
    amenity = choose_amenity(rng)
    furnishing_status = choose_furnishing_status(rng)

    size = round(clipped(rng.gauss(2100, 850), 550, 6200), 2)
    rooms = int(clipped(round(size / rng.uniform(470, 720)), 1, 8))
    bathrooms = int(clipped(round(rooms * rng.uniform(0.45, 0.85)), 1, 6))
    floors = rng.choices([1, 2, 3], weights=[0.45, 0.42, 0.13], k=1)[0]
    parking_spaces = rng.choices([0, 1, 2, 3, 4], weights=[0.12, 0.28, 0.38, 0.16, 0.06], k=1)[0]
    property_age = int(clipped(rng.expovariate(1 / 22), 0, 90))

    distance_base = {
        "Downtown": rng.gauss(3.5, 2.0),
        "Waterfront": rng.gauss(7.0, 3.0),
        "Suburb": rng.gauss(13.0, 5.0),
        "Rural": rng.gauss(29.0, 9.0),
    }[location]
    distance = round(clipped(distance_base, 0.3, 55.0), 2)

    base_price = 55000
    size_component = size * 185
    room_component = rooms * 14500
    bathroom_component = bathrooms * 26000
    floor_component = (floors - 1) * 18000
    parking_component = parking_spaces * 16000
    amenity_component = AMENITIES[amenity]
    furnishing_component = FURNISHING_STATUS[furnishing_status]

    age_penalty = property_age * 2400
    premium_location_age_cushion = property_age * 950 if location in {"Downtown", "Waterfront"} else 0
    distance_penalty = distance * 4200

    expected_price = (
        base_price
        + size_component
        + room_component
        + bathroom_component
        + floor_component
        + parking_component
        + amenity_component
        + furnishing_component
        - age_penalty
        + premium_location_age_cushion
        - distance_penalty
    )
    expected_price *= LOCATIONS[location]

    noise = rng.gauss(0, expected_price * 0.09)
    price = round(clipped(expected_price + noise, 45000, 1800000), 2)

    return {
        "Size": size,
        "NumberOfRooms": rooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "ParkingSpaces": parking_spaces,
        "PropertyAge": property_age,
        "DistanceToCityCenter": distance,
        "Location": location,
        "Amenities": amenity,
        "FurnishingStatus": furnishing_status,
        "Price": price,
    }


def generate_synthetic_dataset(
    output_path: str | Path = DEFAULT_DATA_PATH,
    rows: int = DEFAULT_SYNTHETIC_ROWS,
    random_state: int = RANDOM_STATE,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_state)
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for _ in range(rows):
            writer.writerow(generate_row(rng))

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic house price dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH, help="Where to save the generated CSV.")
    parser.add_argument("--rows", type=int, default=DEFAULT_SYNTHETIC_ROWS, help="Number of rows to generate.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed for reproducible data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_synthetic_dataset(args.output, args.rows, args.random_state)
    print(f"Generated {args.rows} synthetic rows at: {output_path}")


if __name__ == "__main__":
    main()
