# Synthetic Dataset Notes

This project uses generated data for prototype and research purposes.

The default generator is [src/synthetic_data.py](src/synthetic_data.py). It creates `data/house_data.csv` with 5,000 rows unless another row count is provided.

## Pricing Logic

The generated price is based on a deliberately realistic but simplified formula:

- Larger `Size` increases price.
- More rooms, bathrooms, floors, and parking spaces increase price.
- `Downtown` and `Waterfront` locations receive price premiums.
- `Rural` locations receive a lower location multiplier.
- More distance from the city center reduces price.
- Older homes lose value.
- Older premium-location homes lose less value than older non-premium homes.
- Amenities add value:
  - Garden
  - Gym
  - Pool
  - SmartHome
- Furnishing status adds value:
  - Semi-Furnished
  - Fully-Furnished
- Random noise is added to prevent the model from learning a perfect deterministic formula.

## Important Limitation

This is not real market data. It is useful for demonstrating a machine learning workflow, but it should not be used to estimate actual property values.

