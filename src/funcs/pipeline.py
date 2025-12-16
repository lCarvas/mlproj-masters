from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FunctionTransformer, Pipeline

from funcs.custom_transformers import (
    ConstantVarianceRemover,
    FillNATransformer,
    GetDummiesTransformer,
    HighCorrelationRemover,
    ScaleDataTransformer,
)
from funcs.preprocessing import (
    bind_data,
    coalesce_null_columns,
    fix_data,
    fix_models,
    fix_no_brand_models,
    remove_unneeded_floats,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from sklearn.base import BaseEstimator


def build_preprocessing_pipeline(  # noqa: PLR0913
    *,
    metric_features: Sequence[str],
    bool_features: Sequence[str],
    categorical_features: Sequence[str],
    thresholds: Mapping[str, Mapping[Literal["lower", "upper"], float | None]],
    winsorize: bool = False,
    remove_outliers: bool = False,
    unneeded_float_features: Sequence[str],
    scaling_exclude_selector: Iterable | None = None,
    max_len_tolerance: int = 2,
    columns_to_coalesce: Sequence[str],
) -> Pipeline:
    """Build a sklearn Pipeline composed of the custom Polars-based transformers.

    Returns an sklearn.pipeline.Pipeline instance.
    """
    steps: tuple[Any, ...] = (
        (
            "fix_brands",
            FunctionTransformer(
                partial(
                    fix_data,
                    col_name="Brand",
                    col_expr=pl.col("Brand")
                    .str.strip_chars()
                    .str.to_lowercase()
                    .str.replace("^[w]$", "vw"),
                    tags={
                        "toyota",
                        "hyundai",
                        "ford",
                        "mercedes",
                        "opel",
                        "audi",
                        "skoda",
                        "bmw",
                        "vw",
                    },
                ),
            ),
        ),
        (
            "fix_transmission",
            FunctionTransformer(
                partial(
                    fix_data,
                    col_name="transmission",
                    col_expr=pl.col("transmission")
                    .str.strip_chars()
                    .str.to_lowercase(),
                    tags={
                        "manual",
                        "automatic",
                        "semi-auto",
                        "other",
                        "unknown",
                    },
                ),
            ),
        ),
        (
            "fix_fuel_type",
            FunctionTransformer(
                partial(
                    fix_data,
                    col_name="fuelType",
                    col_expr=pl.col("fuelType")
                    .str.strip_chars()
                    .str.to_lowercase(),
                    tags={"petrol", "diesel", "hybrid", "electric", "other"},
                )
            ),
        ),
        (
            "fix_models",
            FunctionTransformer(
                partial(
                    fix_models,
                    max_len_tolerance=max_len_tolerance,
                    is_second_pass=False,
                )
            ),
        ),
        ("fix_no_brand_models", FunctionTransformer(fix_no_brand_models)),
        (
            "fix_models_2nd_pass",
            FunctionTransformer(
                partial(
                    fix_models,
                    max_len_tolerance=max_len_tolerance,
                    is_second_pass=True,
                )
            ),
        ),
        (
            "bind_data",
            FunctionTransformer(
                partial(
                    bind_data,
                    thresholds=thresholds,
                    winsorize=winsorize,
                    remove_outliers=remove_outliers,
                ),
            ),
        ),
        (
            "fill_na",
            FillNATransformer(
                metric_features=metric_features, bool_features=bool_features
            ),
        ),
        (
            "remove_unneeded_floats",
            FunctionTransformer(
                partial(
                    remove_unneeded_floats,
                    unneeded_float_features=unneeded_float_features,
                ),
            ),
        ),
        (
            "get_dummies",
            GetDummiesTransformer(categorical_features=categorical_features),
        ),
        (
            "coalesce_null_unknown",
            FunctionTransformer(
                partial(
                    coalesce_null_columns,
                    columns=columns_to_coalesce,
                )
            ),
        ),
        (
            "scale_data",
            ScaleDataTransformer(exclude_selector=scaling_exclude_selector),
        ),
    )
    return Pipeline(steps)


def build_feature_selection_pipeline(
    threshold: float,
    estimator: BaseEstimator,
    # n_repeats: int = 10,
    # scoring: str | Sequence[str] | None = None,
    importance_threshold: float | None = 0.05,
) -> Pipeline:
    """Build a sklearn Pipeline for feature selection.

    Returns an sklearn.pipeline.Pipeline instance.
    """
    steps: tuple[Any, ...] = (
        ("constant_variance_filter", ConstantVarianceRemover()),
        ("to_pandas", FunctionTransformer(lambda x: x.to_pandas())),
        (
            "high_correlation_filter",
            HighCorrelationRemover(threshold=threshold),
        ),
        # (
        #     "feature_selector",
        #     PermutationImportanceSelector(
        #         estimator=estimator,
        #         n_repeats=n_repeats,
        #         scoring=scoring,
        #         importance_threshold=importance_threshold,
        #     ),
        # ),
        (
            "feature_selector",
            SelectFromModel(
                estimator=estimator, threshold=importance_threshold
            ),
        ),
    )
    return Pipeline(steps)
