import polars as pl
import seaborn as sns
from pandas import DataFrame


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
    return {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}


def describe_data(
    df: pl.DataFrame, metricFeatures: list[str], categoricalFeatures: list[str]
) -> None:
    """Prints the number of duplicated rows and the count and percentage of missing values for each column in the DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame to analyze.
        metricFeatures (list[str]): List of names of metric features.
        categoricalFeatures (list[str]): List of names of categorical features.
    """
    print(f"Duplicated: {df.is_duplicated().sum()}")
    df_len: int = df.shape[0]

    print("Missing: ")
    for col in df.columns:
        null_count = df.get_column(col).null_count()
        print(f"{col}: {null_count}/{df_len} ({null_count / df_len:.2%})")

    for col in categoricalFeatures:
        print(df.get_column(col).value_counts())

    pandas_df: DataFrame = df.to_pandas()

    for i, col in enumerate(metricFeatures):
        sns.boxplot(x=col, data=pandas_df)
