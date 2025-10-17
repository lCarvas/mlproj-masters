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


def fix_models(df: pl.DataFrame) -> pl.DataFrame:
    """Fix model names in the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.

    Returns:
        pl.DataFrame: Polars DataFrame with fixed model names.
    """
    df = df.with_columns(pl.col("model").str.strip_chars().str.to_lowercase())

    df = df.with_columns(
        pl.struct(["model", "Brand"])
        .map_elements(
            lambda x: fix_model_spelling(x["model"], x["Brand"]),
            return_dtype=pl.String,
        )
        .alias("model")
    )

    return df


def fix_model_spelling(element: str, brand: str) -> str:
    """Fix model spelling for a given element.

    Args:
        element (str): The model name to be checked and fixed.
        brand (str): The brand name associated with the model.

    Returns:
        str: The fixed model name, or the original element with "::multiple" or "::none" appended if applicable.
    """
    models: dict[str, tuple[str, ...]] = {
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

    counter: int = 0
    new_model: str = ""

    for model in models.get(brand, ""):
        if element is None:
            continue

        if element in ("viva", "mokka", "verso", "golf", "ka"):
            return element

        if element == "k":
            return "ka"

        if model.startswith(element):
            if len(model) - len(element) <= 2:
                counter += 1
                new_model = model

    if element is None:
        return element

    if brand is None:
        if len(element) == 1:
            return element + "::no_brand"

        if element in ("viva", "mokka", "verso", "golf", "i3", "i8"):
            return element

        no_brand_counter: int = 0

        for model_tuple in models.values():
            for model in model_tuple:
                if model.startswith(element):
                    if len(model) - len(element) <= 2:
                        no_brand_counter += 1
                        no_brand_new_model: str = model

        if no_brand_counter > 1:
            return element + "::multiple::no_brand"
        elif no_brand_counter == 1:
            return no_brand_new_model + "::no_brand"
        else:
            return element + "::none::no_brand"

    if counter > 1:
        return element + "::multiple"
    elif counter == 1:
        return new_model
    else:
        return element + "::none"


def fix_transmission(df: pl.DataFrame) -> pl.DataFrame:
    """Fix transmission types in the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.

    Returns:
        pl.DataFrame: Polars DataFrame with fixed transmission types.
    """
    df = df.with_columns(
        pl.col("transmission")
        .str.strip_chars()
        .str.to_lowercase()
        .map_elements(fix_transmission_spelling)
    )

    return df


def fix_transmission_spelling(element: str) -> str:
    """Fix transmission spelling for a given element.

    Args:
        element (str): The transmission type to be checked and fixed.

    Returns:
        str: The fixed transmission type or the original element with "::none" appended if no match is found.
    """
    transmissions: tuple[str, ...] = (
        "manual",
        "automatic",
        "semi-auto",
        "other",
        "unknown",
    )

    for transmission in transmissions:
        if element in transmission:
            return transmission

    return element + "::none"


def fix_fuel_type(df: pl.DataFrame) -> pl.DataFrame:
    """Fix fuel types in the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame to be modified.

    Returns:
        pl.DataFrame: Polars DataFrame with fixed fuel types.
    """
    df = df.with_columns(
        pl.col("fuelType")
        .str.strip_chars()
        .str.to_lowercase()
        .map_elements(fix_fuel_type_spelling)
    )

    return df


def fix_fuel_type_spelling(element: str) -> str:
    """Fix fuel type spelling for a given element.

    Args:
        element (str): The fuel type to be checked and fixed.

    Returns:
        str: The fixed fuel type or the original element with "::none" appended if no match is found.
    """
    fuel_types: tuple[str, ...] = (
        "petrol",
        "diesel",
        "hybrid",
        "electric",
        "other",
    )

    for fuel_type in fuel_types:
        if element in fuel_type:
            return fuel_type

    return element + "::none"
