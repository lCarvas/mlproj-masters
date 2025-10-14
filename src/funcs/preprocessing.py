from typing import Literal

import polars as pl


def fill_na(
    df: pl.DataFrame, metricFeatures: list[str], boolFeatures: list[str]
) -> pl.DataFrame:
    for col in metricFeatures:
        df = df.with_columns(
            pl.when(pl.col(col).is_null())
            .then(pl.col(col).median())
            .otherwise(pl.col(col))
            .alias(col)
        )

    for col in boolFeatures:
        df = df.with_columns(
            pl.when(pl.col(col).is_null())
            .then(pl.lit(0))
            .otherwise(pl.col(col))
            .alias(col)
        )

    return df


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

