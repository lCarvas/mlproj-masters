import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
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


def corr_heatmap(corr: pd.DataFrame) -> None:
    """Plots a heatmap of the correlation matrix.

    Args:
        corr (pd.DataFrame): The correlation matrix as a Pandas DataFrame.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(data=corr, annot=True, cmap=plt.cm.get_cmap("Reds"), fmt=".1")
    plt.show()

