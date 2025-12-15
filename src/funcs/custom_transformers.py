from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Self

    import pandas as pd
    from numpy.typing import NDArray
    from polars._typing import PythonLiteral
    from sklearn.utils import Bunch


class TransformerBase(BaseEstimator, TransformerMixin):
    """Base transformer class."""


class FillNATransformer(TransformerBase):
    """Tranformer used to fill NA values.

    Atttributes:
        metric_features (tuple[str, ...]): List of metric feature names
        bool_features (tuple[str, ...]): List of boolean feature names
        _medians (dict[str, Any]): Dictionary to store median values for metric features
    """

    metric_features: tuple[str, ...]
    bool_features: tuple[str, ...]
    _medians: dict[str, PythonLiteral | None]

    def __init__(
        self,
        metric_features: Sequence[str],
        bool_features: Sequence[str],
    ) -> None:
        """Initialize the FillNATransformer.

        Args:
            metric_features (Sequence[str]): Sequence of metric feature names
            bool_features (Sequence[str]): Sequence of boolean feature names
        """
        self.metric_features = cast("tuple[str, ...]", metric_features)
        self.bool_features = cast("tuple[str, ...]", bool_features)
        self._medians = {}

    def fit(self, x: pl.DataFrame, _y: pl.Series | None = None) -> Self:
        """Fit the transformer to the data by calculating medians for metric features.

        Args:
            x (pl.DataFrame): Input data.
            _y (pl.Series | None, optional):
                Ignored, present for sklearn compatibility. Defaults to None.

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
        for feat, median in self._medians.items():
            x = x.with_columns(pl.col(feat).fill_null(median))
        for feat in self.bool_features:
            x = x.with_columns(pl.col(feat).fill_null(0))
        return x


class GetDummiesTransformer(TransformerBase):
    """Transformer to convert categorical features into dummy variables.

    Attributes:
        categorical_features (tuple[str]): List of categorical feature names
        _columns (list[str] | None): List of all columns
    """

    categorical_features: tuple[str, ...]
    _columns: list[str] | None

    def __init__(self, categorical_features: Sequence[str]) -> None:
        """Initialize the GetDummiesTransformer.

        Args:
            categorical_features (Sequence[str]):
                Sequence of categorical feature names
        """
        self.categorical_features = cast(
            "tuple[str, ...]", categorical_features
        )
        self._columns = None

    def fit(self, x: pl.DataFrame, _y: pl.Series | None = None) -> Self:
        """Fit the transformer to the data.

        Fit the transformer to the data by determining all possible dummy variable columns.

        Args:
            x (pl.DataFrame): Input data.
            _y (pl.Series | None, optional):
                Ignored, present for sklearn compatibility. Defaults to None.

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
        all_columns: set[str] = set(self._columns or ())
        for col in all_columns:
            if col not in x_dummies.columns:
                x_dummies = x_dummies.with_columns(pl.lit(0).alias(col))
        final_cols: list[str] = sorted(all_columns)
        return x_dummies.select(final_cols)


class ScaleDataTransformer(TransformerBase):
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
                Ignored, present for sklearn compatibility. Defaults to None.

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


class ConstantVarianceRemover(TransformerBase):
    """Transformer to remove features with constant variance.

    Attributes:
        _constant_variance_features (list[str]): List of features with constant variance
    """

    _constant_variance_features: list[str]

    def __init__(self) -> None:
        """Initialize the ConstantVarianceRemover."""
        self._constant_variance_features = []

    def fit(
        self,
        x: pl.DataFrame,
        _y: pl.Series | None = None,
    ) -> Self:
        """Fit the transformer by identifying constant variance features.

        Args:
            x (pl.DataFrame): Input data.
            _y (pl.Series | None, optional):
                Ignored, present for sklearn compatibility. Defaults to None.

        Returns:
            Self: The fitted transformer instance
        """
        for col in x.columns:
            if x.get_column(col).n_unique() <= 1:
                self._constant_variance_features.append(col)
        return self

    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        """Transform the data by removing constant variance features.

        Args:
            x (pl.DataFrame): Input data.

        Returns:
            pl.DataFrame: Transformed data with constant variance features removed.
        """
        return x.select(cs.exclude(self._constant_variance_features))


class HighCorrelationRemover(TransformerBase):
    """Transformer to remove highly correlated features.

    Attributes:
        threshold (float): Correlation threshold for feature removal
        _features_to_remove (list[str]): List of features to be removed
    """

    threshold: float
    _features_to_remove: list[str]

    def __init__(self, threshold: float = 0.8) -> None:
        """Initialize the HighCorrelationRemover.

        Args:
            threshold (float, optional): Correlation threshold for feature removal.
                Defaults to 0.8.
        """
        self.threshold = threshold
        self._features_to_remove = []

    def fit(
        self,
        x: pd.DataFrame,
        _y: pd.Series | None = None,
    ) -> Self:
        """Fit the transformer by identifying highly correlated features.

        Args:
            x (pl.DataFrame): Input data.
            _y (pl.Series | None, optional):
                Ignored, present for sklearn compatibility. Defaults to None.

        Returns:
            Self: The fitted transformer instance
        """
        corr_matrix: pd.DataFrame = x.corr("spearman").abs()
        upper_triangle: pd.DataFrame = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > self.threshold):
                self._features_to_remove.append(column)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by removing highly correlated features.

        Args:
            x (pl.DataFrame): Input data.

        Returns:
            pl.DataFrame: Transformed data with highly correlated features removed.
        """
        return x.loc[:, ~x.columns.isin(self._features_to_remove)]


class PermutationImportanceSelector(TransformerBase):
    """Feature selector using permutation importance.

       Fits the estimator, calculates permutation importance for each feature,
       and selects features with importance above the mean.

    Attributes:
        estimator (BaseEstimator):
            Sklearn estimator to use for calculating feature importance.
        n_repeats (int): Number of times to permute each feature.
        random_state (int): Random state for reproducibility.
        scoring (str | Sequence[str] | None):
            Scoring metric to use. If None, uses estimator's default score.
        importance_threshold (float | None): Threshold used for feature selection.
            If None the mean of feature importances will be used.
        _estimator (BaseEstimator):
            Fitted estimator used for importance calculation.
        _selected_features (list[str]): Boolean mask of selected features.
    """

    estimator: BaseEstimator
    n_repeats: int
    random_state: int
    scoring: str | Sequence[str] | None
    importance_threshold: float | None
    _estimator: BaseEstimator
    _selected_features: NDArray[np.bool]

    def __init__(
        self,
        estimator: BaseEstimator,
        n_repeats: int = 10,
        random_state: int = 42,
        scoring: str | Sequence[str] | None = None,
        importance_threshold: float | None = None,
    ) -> None:
        """Initialize the PermutationImportanceSelector.

        Args:
            estimator (BaseEstimator):
                sklearn estimator to use for calculating feature importance.
            n_repeats (int, optional):
                Number of times to permute each feature. Defaults to 10.
            random_state (int, optional):
                Random state for reproducibility. Defaults to 42.
            scoring (str | Sequence[str] | None, optional):
                Scoring metric to use. If None, uses estimator's default score.
                Defaults to None.
            importance_threshold (float, optional):
                Threshold used for feature selection.
                If None the mean of feature importances will be used.
                Defaults to None.
        """
        self.estimator = estimator
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring
        self.importance_threshold = importance_threshold

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> Self:
        """Fit the transformer by calculating permutation importance.

        Args:
            x (pd.DataFrame): Input data.
            y (pd.Series): Target variable.

        Returns:
            Self: The fitted transformer instance.
        """
        self._estimator = clone(self.estimator)
        self._estimator.fit(x, y)  # pyright: ignore[reportAttributeAccessIssue]

        result: Bunch | dict[str, Bunch] = permutation_importance(
            self._estimator,
            x,
            y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=self.scoring,
        )

        mean_importance: Bunch = result["importances_mean"]

        if self.importance_threshold is None:
            self.importance_threshold = mean_importance.mean()

        self._selected_features = cast(
            "NDArray[np.float64]", mean_importance
        ) >= cast("float", self.importance_threshold)

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by selecting important features.

        Args:
            x (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with selected features.
        """
        return x.loc[:, self._selected_features]
