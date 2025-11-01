from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Self

    from polars._typing import PythonLiteral


class PolarsTransformerBase(BaseEstimator, TransformerMixin):
    """Base transformer that accepts polars/pandas/numpy and returns same type."""


class FillNATransformer(PolarsTransformerBase):
    """Tranformer used to fill NA values.

    Atttributes:
        metric_features (list[str]): List of metric feature names
        bool_features (list[str]): List of boolean feature names
        _medians (dict[str, Any]): Dictionary to store median values for metric features
    """

    metric_features: list[str]
    bool_features: list[str]
    _medians: dict[str, PythonLiteral | None]

    def __init__(
        self, metric_features: Sequence[str], bool_features: Sequence[str]
    ) -> None:
        """Initialize the FillNATransformer.

        Args:
            metric_features (Sequence[str]): Sequence of metric feature names
            bool_features (Sequence[str]): Sequence of boolean feature names
        """
        self.metric_features = list(metric_features)
        self.bool_features = list(bool_features)
        self._medians = {}

    def fit(self, x: pl.DataFrame, _y: pl.Series | None = None) -> Self:
        """Fit the transformer to the data by calculating medians for metric features.

        Args:
            x (pl.DataFrame): Input data.
            _y (pl.Series | None, optional): Target variable. Defaults to None.

        Returns:
            Self: The fitted transformer instance.
        """
        for feat in self.metric_features:
            self._medians[feat] = x.get_column(feat).median()
        return self

    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        """Transform the data by filling NA values.

        Args:
            x (pl.DataFrame): Input data.

        Returns:
            pl.DataFrame: Transformed data with NA values filled.
        """
        # pl_df, orig_type, orig_cols = self._to_polars(x)
        for feat, median in self._medians.items():
            x = x.with_columns(pl.col(feat).fill_null(median))
        for feat in self.bool_features:
            x = x.with_columns(pl.col(feat).fill_null(0))
        return x


class GetDummiesTransformer(PolarsTransformerBase):
    """Transformer to convert categorical features into dummy variables.

    Attributes:
        categorical_features (list[str]): List of categorical feature names
        _columns (list[str] | None): List of all columns
    """

    categorical_features: list[str]
    _columns: list[str] | None

    def __init__(self, categorical_features: Sequence[str]) -> None:
        """Initialize the GetDummiesTransformer.

        Args:
            categorical_features (Sequence[str]):
                Sequence of categorical feature names
        """
        self.categorical_features = list(categorical_features)
        self._columns = None

    def fit(self, x: pl.DataFrame, _y: pl.Series | None = None) -> Self:
        """Fit the transformer to the data.

        Fit the transformer to the data by determining all possible dummy variable columns.

        Args:
            x (pl.DataFrame): Input data.
            _y (pl.Series | None, optional):
                Target Variable Series. Defaults to None.

        Returns:
            Self: The fitted transformer instance.
        """
        x_dummies: pl.DataFrame = x.to_dummies(
            columns=self.categorical_features
        )
        self._columns = sorted(x_dummies.columns)
        return self

    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        """Convert categorical features into dummy variables.

        Args:
            x (pl.DataFrame): Input data.

        Returns:
            pl.DataFrame: Transformed data with dummy variables.
        """
        x_dummies: pl.DataFrame = x.to_dummies(
            columns=self.categorical_features
        )
        # add missing columns with zeros
        all_columns: set[str] = set(self._columns or [])
        for col in all_columns:
            if col not in x_dummies.columns:
                x_dummies = x_dummies.with_columns(pl.lit(0).alias(col))
        final_cols: list[str] = sorted(all_columns)
        return x_dummies.select(final_cols)


class ScaleDataTransformer(PolarsTransformerBase):
    """Transformer to scale data to the range [0, 1].

    Attributes:
        exclude_selector (cs.Selector): Selector to exclude certain columns from scaling
        _mins (dict[str, Any]): Dictionary to store minimum values for each column
        _maxs (dict[str, Any]): Dictionary to store maximum values for each column
    """

    exclude_selector: tuple[str, ...]
    _mins: dict[str, int | float]
    _maxs: dict[str, int | float]

    def __init__(self, exclude_selector: Iterable[str] | None = None) -> None:
        """Initialize the ScaleDataTransformer.

        Args:
            exclude_selector (cs.Selector, optional):
                Selector to exclude certain columns from scaling.
                Defaults to cs.exclude( cs.contains("_"),
                cs.contains("hasDamage"), cs.contains("carID") ).
        """
        self.exclude_selector = (
            tuple(exclude_selector) if exclude_selector is not None else ()
        )
        self._mins = {}
        self._maxs = {}
        self._exclude_selector: cs.Selector = cs.exclude(
            *(cs.contains(p) for p in self.exclude_selector)
        )

    def fit(self, x: pl.DataFrame, _y: pl.Series | None = None) -> Self:
        """Fit the transformer to the data.

        Fit the transformer to the data by calculating min and max for each column.

        Args:
            x (pl.DataFrame): Input data.
            _y (pl.Series | None, optional):
                Target Variable Series. Defaults to None.

        Returns:
            Self: The fitted transformer instance.
        """
        mins: dict[str, int | float] = (
            x.select(self._exclude_selector).min().to_dicts()[0]
        )

        maxs: dict[str, int | float] = (
            x.select(self._exclude_selector).max().to_dicts()[0]
        )

        self._mins = mins
        self._maxs = maxs
        return self

    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        """Transform the data by scaling it to the range [0, 1].

        Args:
            x (pl.DataFrame): Input data.

        Returns:
            pl.DataFrame: Transformed data with scaled values.
        """
        return x.with_columns(
            [
                (pl.col(col) - self._mins[col])
                / (self._maxs[col] - self._mins[col])
                for col in x.select(self._exclude_selector).columns
            ]
        )
