# Notes

## Exploration

### Variables

- carID (index): Int64
- Brand (Categorical) : String
- model (Categorical): String
- year (Categorical): Float64, should be Int64
- price (Metric): Int64
- transmission (Categorical): String
- mileage (Metric): Float64, should be Int64
- fuelType (Categorical): String
- tax (Metric): Float64, should probably be Int64, should not contain negative
  values
- mpg (Metric): Float64, contains floats with differing amounts of decimal
  places, should standardise to 1 decimal place
- engineSize (Metric): Float64, contains floats with differing amounts of
  decimal places, should standardise to 1 decimal place
- paintQuality (Metric)%: Float64, should be Int64
- previousOwners (Metric): Float64, should probably be Int64, should not contain
  negative values
- hasDamage (Categorical, Boolean): Float64, should be Int64

Metric Features: ["price", "mileage", "tax", "mpg", "engineSize",
"paintQuality", "previousOwners"]

Categorical Features: ["Brand", "model", "year", "transmission", "fuelType",
"hasDamage"]

### Duplicated Values

0 rows

### Missing Values

- carID: 0/75973 (0.00%)
- Brand: 1521/75973 (2.00%)
- model: 1517/75973 (2.00%)
- year: 1491/75973 (1.96%)
- price: 0/75973 (0.00%)
- transmission: 1522/75973 (2.00%)
- mileage: 1463/75973 (1.93%)
- fuelType: 1511/75973 (1.99%)
- tax: 7904/75973 (10.40%)
- mpg: 7926/75973 (10.43%)
- engineSize: 1516/75973 (2.00%)
- paintQuality%: 1524/75973 (2.01%)
- previousOwners: 1550/75973 (2.04%)
- hasDamage: 1548/75973 (2.04%)
