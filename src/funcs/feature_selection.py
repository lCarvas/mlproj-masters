import pandas as pd
import polars as pl
from IPython.display import display


def check_variance(df: pl.DataFrame) -> None:
    """Prints the variance of each numeric column in the DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame to analyze.
    """
    for col in df.columns:
        current_column: pl.Series = df.get_column(col)
        if current_column.dtype == pl.String:
            continue
        display(f"{col}: {current_column.var()}")


def get_correlation(df: pl.DataFrame) -> pd.DataFrame:
    """Calculates the correlation matrix of the DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame to analyze.

    Returns:
        pl.DataFrame: A Polars DataFrame representing the correlation matrix.
    """
    return df.to_pandas().corr()

