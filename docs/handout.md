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

Putting the obvious change that was made to the dataset to fix the first point
mentioned above aside, to tackle the unrealistic values we decided to bind the
data to values that made sense.

The values that we decided on are as follows:

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
    leaving them in would negatively impact the performance of our model, as
    such, this upper breakpoint is subject to change.
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

Filling with 0. The only boolean feature present in the dataset is the hasDamage
feature. We can assume that if a car was checked by a mechanic, and was damaged,
it would not have been forgotten. As such filling with 0 did not seem
unreasonable. Even without this assumption, every non-missing value is 0, and,
with only 2.04% of missing values, we can safely assume that these missing
values are much more likely to be 0 than 1.

#### Categorical Features

Currently doing nothing to these missing values. In the future, we want to try
filling with "Unknown" or the mode of the feature.

### Issues in Categorical Features

The columns for the brand, model, transmission and fuel type of the car present
spelling mistakes. Fixing brand, transmission and fuel type is easy, as the
correct values are easy to discover. We just have to compare each of the values
in these columns to the correct values and replace them with the appropriate,
corrected value.

The model column however, was harder to fix. There is a lot of overlap between
models of different brands, for example the i3 from BMW and the I30 from
Hyundai, if we are presented with a model i3, should we leave it as i3, or
should we replace it by i30? Just getting the list of corrected models was not
enough, we had to look for a list of models for each of the brands present in
the dataset. After acquiring this list, we fixed the values by comparing the
current value to the ones in the list, taking into account the brand of the
entry (if it is not missing), this way, a i3 entry with brand BMW will remain as
i3 whilst a i3 entry with brand Hyundai will be replaced by i30.

There's also the situation on which we want to fix a model, but don't have the
brand of an entry, in this case, we first got the brand of the entry by
comparing its model to the aforementioned list of models, getting the brand that
the model belongs to, then running the function to fix models once more.

Thus the process is as follows:

- Run `fix models` to fix all models possible
- Run `fix models with no brand`
- Run `fix models` again to fix more models, now that brands is filled

For this process to run as smoothly as possible, some assumptions had to be
made, they are as follows:

1. If the brand is missing, models `i3` and `i8` belong to BMW. This was
   concluded by manually looking into the dataset, and comparing the remaining
   features.
2. No more than 2 characters were removed in models. This avoids some cases of
   multiple matches, for example, if there was an entry `cl`, it will be
   replaced by `clk` and not `cl class`, `cla class` or `clc class` due to these
   being much longer than the entry present in the dataset.
3. For models that have exactly 2 matches we decided to maintain the value
   present in the dataset instead of trying to replace it. This would lead to an
   entry having multiple matches, which could only be resolvable by comparing
   the remaining features of the entry to those of the possible matches. We
   decided against this approach, simply because keeping the original entry was
   easier. An example of this are the models `ka` and `ka+` and `mokka` and
   `mokka x`.
4. Model entries `a` and `q` belong to Audi. Model entries `x` belong to BMW.
   Any model that starts with these letters besides the ones that belong to the
   aforementioned brands have a character difference larger than 2, as such,
   they're invalidated by assumption 2.
5. Model entries `k` are considered `ka` and not `ka+`. The features of these
   models are extremely similar, therefore, whether we assign `k` to `ka` or
   `ka+` shouldn't really matter.

### Dummy Variables

Currently we are making n dummies for each categorical feature instead of n-1, n
being the number of different values existent in that feature. This is done to
ensure that all columns of the training set appear in the test set, we are aware
this introduces multicollinearity.

### Data Scaling

We decided to go with min-max scaling.

As we're using polars, we wanted to make use of our own methods as much as
possible to avoid excessive conversion to pandas dataframes/series or numpy
arrays. Min-max scaling was the one that came up to discussion first and as such
we decided to implement it.

## Feature Selection

Currently we are only using the
[SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
object from sklearn with the
[f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
scoring function with the number of features to select as 100. This value is not
representative of the best number of features for our model, it's a temporary
number we selected. This is a temporary step until we develop a better way of
automating feature selection and integrating it with sklearn's pipelines.

## Model

As we're tackling a regression problem, for our first model we decided to go
with a
[Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

### Model Selection

We decided to go with the hold-out method, chosen due to its simplicity to
implement.

Currently, to evaluate model performance we're only using the MAE.
