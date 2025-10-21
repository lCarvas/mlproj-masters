from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from numpy import dtype, ndarray
    from scipy.sparse._csr import csr_matrix


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


def rfe(
    x_train: pl.DataFrame,
    x_val: pl.DataFrame,
    y_train: pl.Series,
    y_val: pl.Series,
    model: LinearRegression,
) -> None:
    """Performs Recursive Feature Elimination (RFE) to select the optimal number of features.

    Args:
        x_train (pl.DataFrame): Train DataFrame containing regressor features.
        x_val (pl.DataFrame): Validation DataFrame containing regressor features.
        y_train (pl.Series): Train Series containing target variable.
        y_val (pl.Series): Validation Series containing target variable.
        model (LinearRegression): The model to use for RFE.
    """
    high_score: float = 0
    features_to_select: pd.Series = pd.Series()
    nof: int = 0

    for n in range(len(x_train.columns)):
        rfe = RFE(estimator=model, n_features_to_select=n + 1)

        x_train_rfe: ndarray[tuple[Any, ...], dtype[Any]] = rfe.fit_transform(
            x_train.to_pandas(), y_train.to_pandas()
        )
        x_val_rfe: csr_matrix | ndarray[tuple[Any, ...], dtype[Any]] = (
            rfe.transform(x_val.to_pandas())
        )

        y_train_pandas: pd.Series[int] = y_train.to_pandas()

        model.fit(x_train_rfe, y_train_pandas)

        val_score: float = float(model.score(x_val_rfe, y_val.to_pandas()))

        if val_score >= high_score:
            high_score = val_score
            nof = n + 1

            features_to_select = pd.Series(rfe.support_, index=x_train.columns)

    display(f"Optimal number of features: {nof}")
    display(f"Score with {nof} features: {high_score}")
    display(f"Features to select:\n{features_to_select}")

