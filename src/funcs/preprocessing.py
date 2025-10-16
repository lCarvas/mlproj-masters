from typing import Literal

import polars as pl
from polars._typing import PythonLiteral


def fill_na(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    metricFeatures: list[str],
    boolFeatures: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fills NA values in the DataFrame.

    Args:
        train_df (pl.DataFrame): Train Polars DataFrame to fill NA values.
        test_df (pl.DataFrame): Validation Polars DataFrame to fill NA values.
        metricFeatures (list[str]): Metric features to fill NA with median.
        boolFeatures (list[str]): Boolean features to fill NA with 0.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the modified train and validation DataFrames.
    """
    # train_df = train_df.with_columns(
    #     pl.when(pl.col(col).is_null())
    #     .then(train_col_median)
    #     .otherwise(pl.col(col))
    #     .alias(col)
    # )

    # test_df = test_df.with_columns(
    #     pl.when(pl.col(col).is_null())
    #     .then(train_col_median)
    #     .otherwise(pl.col(col))
    #     .alias(col)
    # )

    # train_df = train_df.with_columns(
    #     pl.when(pl.col(feat).is_null())
    #     .then(pl.lit(0))
    #     .otherwise(pl.col(feat))
    #     .alias(feat)
    # )

    # test_df = test_df.with_columns(
    #     pl.when(pl.col(feat).is_null())
    #     .then(pl.lit(0))
    #     .otherwise(pl.col(feat))
    #     .alias(feat)
    # )

    for feat in metricFeatures:
        train_col_median: PythonLiteral | None = train_df.get_column(
            feat
        ).median()

        train_df = train_df.with_columns(
            pl.col(feat).fill_null(train_col_median)
        )

        test_df = test_df.with_columns(pl.col(feat).fill_null(train_col_median))

    for feat in boolFeatures:
        train_df = train_df.with_columns(pl.col(feat).fill_null(0))
        test_df = test_df.with_columns(pl.col(feat).fill_null(0))

    return train_df, test_df


def bind_data(
    df: pl.DataFrame,
    thresholds: dict[str, dict[Literal["lower", "upper"], float | None]],
) -> pl.DataFrame:
    """Bind data within specified thresholds.

    Args:
        df (pl.DataFrame): Polars DataFrame to be filtered.
        thresholds (dict[str, dict[Literal["lower", "upper"], float | None]]):
            A dictionary where keys are column names and values are dictionaries
            with 'lower' and 'upper' keys specifying the threshold values.

    Returns:
        pl.DataFrame: Filtered Polars DataFrame.
    """
    for k, v in thresholds.items():
        if v["lower"] is not None:
            df = df.filter(pl.col(k) >= v["lower"])

        if v["upper"] is not None:
            df = df.filter(pl.col(k) <= v["upper"])

    return df


def remove_unneeded_floats(
    df: pl.DataFrame, unneeded_float_features: list[str]
) -> pl.DataFrame:
    """Convert specified float columns to integers.

    This only strips the decimal part of the float, it does not round the values.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.
        unneeded_float_features (list[str]): List of column names to convert from float to int.

    Returns:
        pl.DataFrame: Polars DataFrame with specified columns converted to integers.
    """
    for col in unneeded_float_features:
        df = df.with_columns(pl.col(col).cast(pl.Int64))
    return df


def remove_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Remove duplicate rows from the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.

    Returns:
        pl.DataFrame: Polars DataFrame with duplicate rows removed.
    """
    return df.unique()


def get_dummies(
    df: pl.DataFrame, categorical_features: list[str]
) -> pl.DataFrame:
    """Convert categorical features to dummy variables.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.
        categorical_features (list[str]): List of categorical feature names to convert.

    Returns:
        pl.DataFrame: Polars DataFrame with categorical features converted to dummy variables.
    """
    return df.to_dummies(columns=categorical_features, drop_first=True)


def scale_data(
    df_train: pl.DataFrame, df_test: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Scale numerical features in the DataFrame using Min-Max scaling.

    Args:
        df_train (pl.DataFrame): Train Polars DataFrame to be modified.
        df_test (pl.DataFrame): Validation Polars DataFrame to be modified.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the modified train and validation DataFrames.
    """
    df_train_min = df_train.min().to_dict()
    df_train_max = df_test.max().to_dict()

    df_train = df_train.with_columns(
        [
            (pl.col(col) - df_train_min[col])
            / (df_train_max[col] - df_train_min[col])
            for col in df_train.columns
        ]
    )

    df_test = df_test.with_columns(
        [
            (pl.col(col) - df_train_min[col])
            / (df_train_max[col] - df_train_min[col])
            for col in df_test.columns
        ]
    )

    return df_train, df_test


def fix_brands(df: pl.DataFrame) -> pl.DataFrame:
    """Fix brand names in the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.

    Returns:
        pl.DataFrame: Polars DataFrame with fixed brand names.
    """
    df = df.with_columns(
        pl.col("Brand")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace("^[w]$", "vw")
        .map_elements(fix_brand_spelling)
    )

    return df


def fix_brand_spelling(element: str) -> str:
    """Fix brand spelling for a given element.

    Args:
        element (str): The brand name to be checked and fixed.

    Returns:
        str: The fixed brand name or the original element with "::none" appended if no match is found.
    """
    brands: tuple[str, ...] = (
        "toyota",
        "hyundai",
        "ford",
        "mercedes",
        "opel",
        "audi",
        "skoda",
        "bmw",
        "vw",
    )

    for brand in brands:
        if element in brand:
            return brand

    return element + "::none"
