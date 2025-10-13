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


def removeOutliers(
    df: pl.DataFrame,
    metricFeatures: list[str],
    thresholds: dict[str, dict[Literal["lower", "upper"], float | None]],
) -> pl.DataFrame:
    return df
