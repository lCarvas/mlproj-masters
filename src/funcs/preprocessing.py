from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Literal

import polars as pl
import polars.selectors as cs

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from polars._typing import PythonLiteral

_MODELS: dict[str, tuple[str, ...]] = {
    "audi": (
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "q2",
        "q3",
        "q5",
        "q7",
        "q8",
        "r8",
        "rs3",
        "rs4",
        "rs5",
        "rs6",
        "rs7",
        "s3",
        "s4",
        "s5",
        "s8",
        "sq5",
        "sq7",
        "tt",
    ),
    "bmw": (
        "1 series",
        "2 series",
        "3 series",
        "4 series",
        "5 series",
        "6 series",
        "7 series",
        "8 series",
        "i3",
        "i8",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "z3",
        "z4",
    ),
    "ford": (
        "b-max",
        "c-max",
        "ecosport",
        "edge",
        "escort",
        "fiesta",
        "focus",
        "fusion",
        "galaxy",
        "grand c-max",
        "grand tourneo connect",
        "ka",
        "ka+",
        "kuga",
        "mondeo",
        "mustang",
        "puma",
        "ranger",
        "s-max",
        "streetka",
        "tourneo connect",
        "tourneo custom",
        "transit tourneo",
    ),
    "hyundai": (
        "accent",
        "amica",
        "getz",
        "i10",
        "i20",
        "i30",
        "i40",
        "i800",
        "ioniq",
        "ix20",
        "ix35",
        "kona",
        "santa fe",
        "terracan",
        "tucson",
        "veloster",
    ),
    "mercedes": (
        "180",
        "200",
        "220",
        "230",
        "a class",
        "b class",
        "c class",
        "cl class",
        "cla class",
        "clc class",
        "clk",
        "cls class",
        "e class",
        "g class",
        "gl class",
        "gla class",
        "glb class",
        "glc class",
        "gle class",
        "gls class",
        "m class",
        "r class",
        "s class",
        "sl class",
        "slk",
        "v class",
        "x-class",
    ),
    "skoda": (
        "citigo",
        "fabia",
        "kamiq",
        "karoq",
        "kodiaq",
        "octavia",
        "rapid",
        "roomster",
        "scala",
        "superb",
        "yeti",
        "yeti outdoor",
    ),
    "opel": (
        "adam",
        "agila",
        "ampera",
        "antara",
        "astra",
        "cascada",
        "combo life",
        "corsa",
        "crossland x",
        "grandland x",
        "gtc",
        "insignia",
        "kadjar",
        "meriva",
        "mokka",
        "mokka x",
        "tigra",
        "vectra",
        "viva",
        "vivaro",
        "zafira",
        "zafira tourer",
    ),
    "vw": (
        "amarok",
        "arteon",
        "beetle",
        "caddy",
        "caddy life",
        "caddy maxi",
        "caddy maxi life",
        "california",
        "caravelle",
        "cc",
        "eos",
        "fox",
        "golf",
        "golf sv",
        "jetta",
        "passat",
        "polo",
        "scirocco",
        "sharan",
        "shuttle",
        "t-cross",
        "t-roc",
        "tiguan",
        "tiguan allspace",
        "touareg",
        "touran",
        "up",
    ),
    "toyota": (
        "auris",
        "avensis",
        "aygo",
        "c-hr",
        "camry",
        "corolla",
        "gt86",
        "hilux",
        "iq",
        "land cruiser",
        "prius",
        "proace verso",
        "rav4",
        "supra",
        "urban cruiser",
        "verso",
        "verso-s",
        "yaris",
    ),
}


def fill_na(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    metric_features: Sequence[str],
    bool_features: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fills NA values in the DataFrame.

    Args:
        train_df (pl.DataFrame): Train Polars DataFrame to fill NA values.
        test_df (pl.DataFrame): Validation Polars DataFrame to fill NA values.
        metric_features (Sequence[str]): Metric features to fill NA with median.
        bool_features (Sequence[str]): Boolean features to fill NA with 0.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]:
            Tuple containing the modified train and validation DataFrames.
    """
    for feat in metric_features:
        train_col_median: PythonLiteral | None = train_df.get_column(
            feat
        ).median()
        train_df = train_df.with_columns(
            pl.col(feat).fill_null(train_col_median)
        )

        test_df = test_df.with_columns(pl.col(feat).fill_null(train_col_median))

    for feat in bool_features:
        train_df = train_df.with_columns(pl.col(feat).fill_null(0))
        test_df = test_df.with_columns(pl.col(feat).fill_null(0))

    return train_df, test_df


def bind_data(
    df: pl.DataFrame,
    thresholds: Mapping[str, Mapping[Literal["lower", "upper"], float | None]],
    *,
    winsorize: bool = False,
    remove_outliers: bool = False,
) -> pl.DataFrame:
    """Bind data within specified thresholds.

    Sets values outside thresholds to None, or winsorizes/removes them.

    Args:
        df (pl.DataFrame): Polars DataFrame to be filtered.
        thresholds (Mapping[str, dict[Literal["lower", "upper"], float | None]]):
            A dictionary where keys are column names and values are dictionaries
            with 'lower' and 'upper' keys specifying the threshold values.
        winsorize (bool, optional):
            If True, values outside thresholds are set to the threshold values.
            Defaults to False.
        remove_outliers (bool, optional):
            If True, rows with values outside thresholds are removed.
            Defaults to False.

    Returns:
        pl.DataFrame: Filtered Polars DataFrame.
    """
    if winsorize:
        return _winsorize_bind(df, thresholds)
    if remove_outliers:
        return _remove_outlier_rows_bind(df, thresholds)
    return _null_outside_bind(df, thresholds)


def _winsorize_bind(
    df: pl.DataFrame,
    thresholds: Mapping[str, Mapping[Literal["lower", "upper"], float | None]],
) -> pl.DataFrame:
    for k, v in thresholds.items():
        if v["lower"] is not None:
            df = df.with_columns(
                pl.when(pl.col(k) < v["lower"])
                .then(v["lower"])
                .otherwise(pl.col(k))
                .alias(k)
            )

        if v["upper"] is not None:
            df = df.with_columns(
                pl.when(pl.col(k) > v["upper"])
                .then(v["upper"])
                .otherwise(pl.col(k))
                .alias(k)
            )

    return df


def _remove_outlier_rows_bind(
    df: pl.DataFrame,
    thresholds: Mapping[str, Mapping[Literal["lower", "upper"], float | None]],
) -> pl.DataFrame:
    for k, v in thresholds.items():
        if v["lower"] is not None:
            df = df.filter(pl.col(k) >= v["lower"])

        if v["upper"] is not None:
            df = df.filter(pl.col(k) <= v["upper"])

    return df


def _null_outside_bind(
    df: pl.DataFrame,
    thresholds: Mapping[str, Mapping[Literal["lower", "upper"], float | None]],
) -> pl.DataFrame:
    for k, v in thresholds.items():
        if v["lower"] is not None:
            df = df.with_columns(
                pl.when(pl.col(k) < v["lower"])
                .then(None)
                .otherwise(pl.col(k))
                .alias(k)
            )

        if v["upper"] is not None:
            df = df.with_columns(
                pl.when(pl.col(k) > v["upper"])
                .then(None)
                .otherwise(pl.col(k))
                .alias(k)
            )

    return df


def remove_unneeded_floats(
    df: pl.DataFrame,
    unneeded_float_features: Sequence[str],
) -> pl.DataFrame:
    """Convert specified float columns to integers.

    This strips the decimal part of the float, it does not round the values.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.
        unneeded_float_features (Sequence[str]):
            List of column names to convert from float to int.

    Returns:
        pl.DataFrame:
            Polars DataFrame with specified columns converted to integers.
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
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    categorical_features: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert categorical features to dummy variables.

    Args:
        df_train (pl.DataFrame): Train Polars DataFrame to be modified.
        df_test (pl.DataFrame): Validation Polars DataFrame to be modified.
        categorical_features (Sequence[str]):
            List of categorical feature names to convert.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]:
            Tuple containing the modified train and validation DataFrames.
    """
    df_train = df_train.to_dummies(columns=categorical_features)

    df_test = df_test.to_dummies(columns=categorical_features)

    all_columns: set[str] = set(df_train.columns).union(set(df_test.columns))

    for col in all_columns:
        if col not in df_train.columns:
            df_train = df_train.with_columns(pl.lit(0).alias(col))
        if col not in df_test.columns:
            df_test = df_test.with_columns(pl.lit(0).alias(col))

    final_cols: list[str] = sorted(all_columns)
    df_train = df_train.select(final_cols)
    df_test = df_test.select(final_cols)

    return df_train, df_test


def scale_data(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Scale numerical features in the DataFrame using Min-Max scaling.

    Args:
        df_train (pl.DataFrame): Train Polars DataFrame to be modified.
        df_test (pl.DataFrame): Validation Polars DataFrame to be modified.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the modified train
        and validation DataFrames.
    """
    exclude_selector: cs.Selector = cs.exclude(
        cs.contains("_"), cs.contains("hasDamage"), cs.contains("carID")
    )

    df_train_min: dict[str, float] = (
        df_train.select(exclude_selector).min().to_dicts()[0]
    )
    df_train_max: dict[str, float] = (
        df_test.select(exclude_selector).max().to_dicts()[0]
    )

    df_train = df_train.with_columns(
        [
            (pl.col(col) - df_train_min[col])
            / (df_train_max[col] - df_train_min[col])
            for col in df_train.select(exclude_selector).columns
        ]
    )

    df_test = df_test.with_columns(
        [
            (pl.col(col) - df_train_min[col])
            / (df_train_max[col] - df_train_min[col])
            for col in df_test.select(exclude_selector).columns
        ]
    )

    return df_train, df_test


def fix_data(
    df: pl.DataFrame, col_name: str, col_expr: pl.Expr, tags: set[str]
) -> pl.DataFrame:
    """Generic function to fix data in a column based on tags.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.
        col_name (str): Name of the column to be fixed.
        col_expr (pl.Expr): Polars expression for the column.
        tags (set[str]): Set of tags to check against.

    Returns:
        pl.DataFrame: Polars DataFrame with fixed data in the specified column.
    """
    return df.with_columns(
        pl.coalesce(
            *[
                pl.when(pl.lit(tag).str.contains(col_expr)).then(pl.lit(tag))
                for tag in tags
            ],
            col_expr + pl.lit("::none"),
        ).alias(col_name)
    )


def fix_models(
    df: pl.DataFrame,
    *,
    max_len_tolerance: int = 2,
    is_second_pass: bool = False,
) -> pl.DataFrame:
    """Fix model names in the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.
        max_len_tolerance (int, optional):
            Max len difference between element and actual model. Defaults to 2.
        is_second_pass (bool, optional):
            Whether this is a second pass for fixing models. Defaults to False.

    Returns:
        pl.DataFrame: Polars DataFrame with fixed model names.
    """
    df = df.with_columns(pl.col("model").str.strip_chars().str.to_lowercase())

    return df.with_columns(
        pl.struct(["model", "Brand"])
        .map_elements(
            lambda x: _fix_model_spelling(
                x["model"],
                x["Brand"],
                max_len_tolerance=max_len_tolerance,
                is_second_pass=is_second_pass,
            ),
            return_dtype=pl.String,
        )
        .alias("model")
    )


def _matches_for_sequence(
    models: Iterable[str], element: str, tol: int
) -> list[str]:
    return [
        model
        for model in models
        if model.startswith(element) and len(model) - len(element) <= tol
    ]


def _search_brand_matches(brand: str, element: str, tol: int) -> list[str]:
    return _matches_for_sequence(_MODELS.get(brand, ()), element, tol)


def _search_all_matches(element: str, tol: int) -> list[str]:
    return _matches_for_sequence(
        chain.from_iterable(_MODELS.values()), element, tol
    )


def _resolve_brand(
    element: str, brand: str, tol: int, *, is_second_pass: bool
) -> str:
    matches: list[str] = _search_brand_matches(brand, element, tol)

    if len(matches) > 1:
        if element in {"viva", "mokka", "verso", "golf", "ka"}:
            return element
        if is_second_pass:
            return "unknown"
        return element + "::multiple"
    if len(matches) == 1:
        return matches[0]
    if is_second_pass:
        return "unknown"
    return element + "::none"


def _resolve_no_brand(element: str, tol: int) -> str:
    if len(element) == 1:
        return element + "::no_brand"

    if element in {"viva", "mokka", "verso", "golf", "ka", "i3", "i8"}:
        return element

    all_matches: list[str] = _search_all_matches(element, tol)

    if len(all_matches) > 1:
        return element + "::multiple::no_brand"
    if len(all_matches) == 1:
        return all_matches[0] + "::no_brand"
    return element + "::none::no_brand"


def _fix_model_spelling(
    element: str | None,
    brand: str | None,
    *,
    max_len_tolerance: int = 2,
    is_second_pass: bool,
) -> str | None:
    """Fix model spelling for a given element and brand.

    Args:
        element (str): The model name to be checked and fixed.
        brand (str | None): The brand name associated with the element.
        max_len_tolerance (int, optional):
            Max len difference between element and actual model. Defaults to 2.
        is_second_pass (bool): Whether this is a second pass for fixing models.

    Returns:
        str: The fixed model name or the original element with appropriate
        suffixes if applicable.
    """
    if element is None:
        return element

    if element == "k":
        element = "ka"

    if brand:
        return _resolve_brand(
            element, brand, max_len_tolerance, is_second_pass=is_second_pass
        )

    return _resolve_no_brand(element, max_len_tolerance)


def fix_no_brand_models(df: pl.DataFrame) -> pl.DataFrame:
    """Fix models with no brand in the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.

    Returns:
        pl.DataFrame: Polars DataFrame with fixed models that had no brand.
    """
    df = df.with_columns(pl.col("model").str.replace("::.*", ""))

    model_names: list[str] = []
    model_brands: list[str] = []
    for brand, models in _MODELS.items():
        for model in models:
            model_names.append(model)
            model_brands.append(brand)

    map_df = pl.DataFrame(
        {"model": model_names, "brand_from_model": model_brands}
    )

    df = df.join(map_df, on="model", how="left")

    df = df.with_columns(
        pl.coalesce([pl.col("Brand"), pl.col("brand_from_model")]).alias(
            "Brand"
        )
    )

    df = df.with_columns(
        pl.when(pl.col("model").str.contains("^x$"))
        .then(pl.lit("bmw"))
        .when(pl.col("model").str.contains("^[aq]$"))
        .then(pl.lit("audi"))
        .otherwise(pl.col("Brand"))
        .alias("Brand")
    )

    return df.drop("brand_from_model")


def coalesce_null_columns(
    df: pl.DataFrame,
    columns: Sequence[str],
) -> pl.DataFrame:
    """Coalesce "null" and "unknown" columns, taking the first non-null value.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.
        columns (Sequence[str]):
            List of column names to coalesce their null and unknown columns.

    Returns:
        pl.DataFrame: Polars DataFrame with the new coalesced column.
    """
    for col in columns:
        unknown_col: str = f"{col}_unknown"
        null_col: str = f"{col}_null"

        has_unknown: bool = unknown_col in df.columns
        has_null: bool = null_col in df.columns

        if has_unknown and has_null:
            df = df.with_columns(
                pl.coalesce(pl.col(unknown_col), pl.col(null_col)).alias(
                    unknown_col
                )
            )
            df = df.drop(null_col)
        elif has_null and not has_unknown:
            df = df.rename({f"{col}_null": f"{col}_unknown"})
    return df
