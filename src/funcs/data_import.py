from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from IPython.display import display


def import_data(file_path: str) -> pl.DataFrame:
    """Imports data from a CSV file and returns it as a Polars DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the imported data.
    """
    return pl.scan_csv(file_path).collect()


def column_types(df: pl.DataFrame) -> dict[str, str]:
    """Returns a dictionary with column names and their data types.

    Args:
        df (pl.DataFrame): The Polars DataFrame to analyze.

    Returns:
        dict[str, str]: A dictionary mapping column names to their data types.
    """
    return {
        col: str(dtype)
        for col, dtype in zip(df.columns, df.dtypes, strict=True)
    }


def describe_data(
    df: pl.DataFrame,
    metric_features: list[str],
    categorical_features: list[str],
) -> None:
    """Displays DataFrame info.

    Displays information about the DataFrame and plots metric features as
    histograms and violin + boxplots.

    Args:
    df (pl.DataFrame): The Polars DataFrame to analyze.
    metric_features (list[str]): List of names of metric features.
    categorical_features (list[str]): List of names of categorical features.
    """
    display(df.describe())
    display(df.schema)
    display(f"Duplicated rows: {df.is_duplicated().sum()}")
    for col in categorical_features:
        display(df.get_column(col).value_counts().transpose())
    n_features = len(metric_features)
    ncols = 6
    nrows = (n_features + ncols - 1) // ncols
    _fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows)
    )
    axes = axes.flatten()
    for i, col in enumerate(metric_features):
        axes[i].hist(
            df[col].drop_nulls().to_numpy(),
            bins=30,
            color="skyblue",
            edgecolor="black",
        )
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()

    _fig2, axes2 = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows)
    )
    axes2 = axes2.flatten()
    for i, col in enumerate(metric_features):
        axes2[i].violinplot(df[col].drop_nulls().to_numpy(), vert=False)
        axes2[i].boxplot(df[col].drop_nulls().to_numpy(), vert=False)
        axes2[i].set_title(col)
    plt.tight_layout()
    plt.show()
