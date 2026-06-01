# Data Schema

The generated dataset contains one row per house.

## Input Features

| Column | Type | Description |
| --- | --- | --- |
| `Size` | Numeric | House size in square feet. |
| `NumberOfRooms` | Numeric | Total room count. |
| `Bathrooms` | Numeric | Number of bathrooms. |
| `Floors` | Numeric | Number of floors. |
| `ParkingSpaces` | Numeric | Available parking spaces. |
| `PropertyAge` | Numeric | Age of the property in years. |
| `DistanceToCityCenter` | Numeric | Distance from city center in kilometers. |
| `Location` | Categorical | One of `Rural`, `Suburb`, `Downtown`, or `Waterfront`. |
| `Amenities` | Categorical | One of `None`, `Garden`, `Gym`, `Pool`, or `SmartHome`. |
| `FurnishingStatus` | Categorical | One of `Unfurnished`, `Semi-Furnished`, or `Fully-Furnished`. |

## Target

| Column | Type | Description |
| --- | --- | --- |
| `Price` | Numeric | Synthetic sale price generated from the pricing formula plus random noise. |

## Example Input For Prediction

```csv
Size,NumberOfRooms,Bathrooms,Floors,ParkingSpaces,PropertyAge,DistanceToCityCenter,Location,Amenities,FurnishingStatus
2200,4,2,2,2,12,11.5,Suburb,Garden,Semi-Furnished
```

