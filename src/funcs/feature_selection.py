from pathlib import Path

import polars as pl
from data_import import import_data


def check_variance(df: pl.DataFrame) -> None:
    """Prints the variance of each numeric column in the DataFrame.
    Args:
        df (pl.DataFrame): The Polars DataFrame to analyze.
    """
    for col in df.columns:
        current_column: pl.Series = df.get_column(col)
        if current_column.dtype == pl.String:
            continue
        print(f"{col}: {current_column.var()}")


# check_variance(import_data(f"{Path.cwd()}/src/data/train.csv"))
