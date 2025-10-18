import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
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
    """Returns a dictionary with column names as keys and their data types as values.

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
    """Prints the number of duplicated rows and the count and percentage of missing values for each column in the DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame to analyze.
        metric_features (list[str]): List of names of metric features.
        categorical_features (list[str]): List of names of categorical features.
    """
    display(f"Duplicated: {df.is_duplicated().sum()}")
    df_len: int = df.shape[0]

    display("Missing: ")
    for col in df.columns:
        null_count = df.get_column(col).null_count()
        display(f"{col}: {null_count}/{df_len} ({null_count / df_len:.2%})")

    for col in categorical_features:
        display(df.get_column(col).value_counts())

    for i, col in enumerate(metric_features):
        plt.figure(i)
        sns.boxplot(x=col, data=df)
