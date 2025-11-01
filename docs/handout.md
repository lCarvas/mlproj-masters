# Machine Learning (200179) Project Handout 2025/2026

## Group Members

- Diogo Carvalho - 20221935
- Luiza Salum - 20221902
- Ricardo Pereira - 20250343

## Preprocessing

### Inconsistencies and Outliers

Throughout the dataset, some values that don't make sense appear, we found the
following:

- Floats in the year, mileage, tax, paintQuality%, previousOwners and hasDamage
  columns, these should be integers.
- Negative values in mileage, tax, mpg, engineSize and previousOwners, these
  should always be positive.
- Unrealistic values such as engine size or miles per gallon being 0. More on
  that below.

Putting the obvious changes that were made to the dataset to fix the first two
aforementioned points, to tackle the unrealistic values we decided to bind data
in these columns to values that made sense, the value that we decided on are as
follows:

- Mileage:
  - Minimum: 1, since Cars 4 You is a resale company, the cars have to be used.
  - Maximum: None, there should be no upper limit to the mileage in a car, if
    anything, this should affect the quality of the car and, consequently, its
    price.
- Tax:
  - Minimum: 0, tax exemptions
  - Maximum: 400, values of tax above 360-370 are outliers, the next value found
    is above 500, 400 was chosen due to it being a nice, round number. Even
    though the outliers don't seem to be unrealistic, we were worried that
    leaving them in would detriment the performance of our model, as such, this
    upper breakpoint is subject to change.
- Miles per Gallon:
  - Minimum: 20, after doing some research, this seemed like a reasonable
    minimum.
  - Maximum: 200, same reasoning as above.
- Engine Size:
  - Minimum: 1.0, same reasoning as miles per gallon
  - Maximum: 6.0, same reasoning as miles per gallon
- Paint Quality %:
  - Minimum: 0, it's a percentage value, there should not be values under 0
  - Maximum: 100, same reasoning as above, there should not be values above 100
- Previous Owners:
  - Minimum: 0, a car cannot have a negative amount of previous owners
  - Maximum: None, considered a maximum of 4, due to there being outliers above
    this value. However, it's not a nonsensical value, so we left it for now,
    just like tax, this is subject to change.

### Missing Values

#### Metric Features

Filling with the median of the respective feature. There was no real reasoning
in this decision, as such, it is very subject to change once a more careful
analysis is done.

#### Boolean Features

Filling with 0. The only boolean feature present in the dataset is the
hasDamage. We can assume that if a car was checked by a mechanic, and had
damage, it would not have been forgotten. As such filling with 0 did not seem
unreasonable. Even without this assumption, every non-missing value is 0, and,
with only 2.04% of missing values, we can safely assume that these missing
values are much more likely to be 0 than 1.

#### Categorical Features

Currently doing nothing to these missing values. However, we want to first try
filling with "Unknown". Whilst we could go with the mode of the feature, we felt
it would be a safer approach to fill with unknown. We have not compared the
performance of the model with these changes to the performance of the current
model so we have yet to decide on how to proceed.

### Issues in Categorical Features
