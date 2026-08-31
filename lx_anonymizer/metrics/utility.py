"""Distribution-level utility metrics for candidate release tables.

Categorical features use base-2 Jensen--Shannon divergence, which lies in
``[0, 1]``. Continuous and ordinal features use the empirical one-dimensional
1-Wasserstein distance. Wasserstein values can be divided by a data range or an
explicit ground-metric scale so heterogeneous features can be aggregated.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

type OrdinalValue = str | int | float


class FeatureType(str, Enum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"


class MissingValuePolicy(str, Enum):
    """Treatment of nulls and configured suppression symbols.

    ``AUTO`` retains them as one category for categorical features and excludes
    them from continuous/ordinal features.
    """

    AUTO = "auto"
    AS_CATEGORY = "as_category"
    EXCLUDE = "exclude"
    ERROR = "error"


class NormalizationStrategy(str, Enum):
    """Scale used for Wasserstein distances."""

    AUTO = "auto"
    NONE = "none"
    COMBINED_RANGE = "combined_range"
    REFERENCE_RANGE = "reference_range"
    FIXED = "fixed"


@dataclass(frozen=True, slots=True)
class UtilityFeatureConfig:
    """Configuration for one release-table feature."""

    name: str
    feature_type: FeatureType
    weight: float
    normalization: NormalizationStrategy = NormalizationStrategy.AUTO
    ground_metric_scale: float | None = None
    ordinal_order: tuple[OrdinalValue, ...] = ()
    missing_policy: MissingValuePolicy = MissingValuePolicy.AUTO
    suppressed_values: tuple[str, ...] = ("*", "SUPPRESSED")
    smoothing: float = 0.0
    log_base: float = 2.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature name must not be blank")
        _require_finite_nonnegative(self.weight, "feature weight")
        _require_finite_nonnegative(self.smoothing, "smoothing")
        if (
            not math.isfinite(self.log_base)
            or self.log_base <= 0.0
            or math.isclose(self.log_base, 1.0)
        ):
            raise ValueError("log_base must be finite, positive, and not equal to 1")
        if self.feature_type is FeatureType.CATEGORICAL:
            if self.normalization not in {
                NormalizationStrategy.AUTO,
                NormalizationStrategy.NONE,
            }:
                raise ValueError("categorical JSD does not accept range normalization")
            if self.ordinal_order:
                raise ValueError("ordinal_order is only valid for ordinal features")
        if self.feature_type is FeatureType.CONTINUOUS and self.ordinal_order:
            raise ValueError("ordinal_order is only valid for ordinal features")
        if self.ordinal_order and len(
            {_category_key(v) for v in self.ordinal_order}
        ) != len(self.ordinal_order):
            raise ValueError("ordinal_order values must be unique")
        if self.normalization is NormalizationStrategy.FIXED:
            if self.ground_metric_scale is None:
                raise ValueError("fixed normalization requires ground_metric_scale")
            _require_finite_positive(self.ground_metric_scale, "ground_metric_scale")
        elif self.ground_metric_scale is not None:
            raise ValueError("ground_metric_scale requires normalization='fixed'")
        if (
            self.feature_type is not FeatureType.CATEGORICAL
            and self.missing_policy is MissingValuePolicy.AS_CATEGORY
        ):
            raise ValueError(
                "as_category missing policy is only valid for categorical features"
            )


@dataclass(frozen=True, slots=True)
class UtilityMetricConfig:
    """Validated configuration for aggregate utility discrepancy."""

    features: tuple[UtilityFeatureConfig, ...]
    tau_max: float | None = None

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("at least one feature must be configured")
        names = [feature.name for feature in self.features]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        total_weight = math.fsum(feature.weight for feature in self.features)
        if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("feature weights must sum to 1")
        if self.tau_max is not None:
            _require_finite_nonnegative(self.tau_max, "tau_max")


@dataclass(frozen=True, slots=True)
class UtilityResult:
    """Feature distances, weighted contributions, and threshold audit result."""

    distance: float
    feature_distances: Mapping[str, float]
    weighted_contributions: Mapping[str, float]
    normalization_scales: Mapping[str, float | None]
    tau_max: float | None = None
    passes_threshold: bool | None = None

    @property
    def d_util(self) -> float:
        return self.distance

    @property
    def within_threshold(self) -> bool | None:
        return self.passes_threshold


_MISSING_CATEGORY = object()


def jensen_shannon_divergence(
    original: Iterable[object],
    release: Iterable[object],
    *,
    smoothing: float = 0.0,
    log_base: float = 2.0,
    missing_policy: MissingValuePolicy = MissingValuePolicy.AS_CATEGORY,
    suppressed_values: Iterable[str] = ("*", "SUPPRESSED"),
) -> float:
    """Return JSD between two empirical categorical distributions.

    Zero-probability terms are omitted mathematically, so smoothing is optional.
    When supplied, ``smoothing`` is an additive pseudocount per observed category.
    """

    _require_finite_nonnegative(smoothing, "smoothing")
    if not math.isfinite(log_base) or log_base <= 0.0 or math.isclose(log_base, 1.0):
        raise ValueError("log_base must be finite, positive, and not equal to 1")
    if missing_policy is MissingValuePolicy.AUTO:
        missing_policy = MissingValuePolicy.AS_CATEGORY
    suppression_set = frozenset(suppressed_values)
    original_values = _categorical_values(
        original, missing_policy=missing_policy, suppressed_values=suppression_set
    )
    release_values = _categorical_values(
        release, missing_policy=missing_policy, suppressed_values=suppression_set
    )
    if not original_values or not release_values:
        raise ValueError(
            "categorical distributions must each contain at least one value"
        )

    original_counts = Counter(original_values)
    release_counts = Counter(release_values)
    categories = tuple(original_counts.keys() | release_counts.keys())
    original_probabilities = _probabilities(
        original_counts, categories, smoothing=smoothing
    )
    release_probabilities = _probabilities(
        release_counts, categories, smoothing=smoothing
    )
    midpoint = (original_probabilities + release_probabilities) / 2.0
    divergence = 0.5 * _kl_divergence(
        original_probabilities, midpoint, log_base
    ) + 0.5 * _kl_divergence(release_probabilities, midpoint, log_base)
    return max(0.0, float(divergence))


def wasserstein_distance(
    original: Iterable[float],
    release: Iterable[float],
    *,
    normalization_scale: float | None = None,
) -> float:
    """Return empirical one-dimensional W1, optionally in ground-metric units."""

    original_values = _finite_float_array(original, "original")
    release_values = _finite_float_array(release, "release")
    if normalization_scale is not None:
        _require_finite_positive(normalization_scale, "normalization_scale")

    all_values = np.concatenate((original_values, release_values))
    all_values.sort()
    deltas = np.diff(all_values)
    if deltas.size == 0:
        distance = 0.0
    else:
        original_cdf = (
            np.searchsorted(original_values, all_values[:-1], side="right")
            / original_values.size
        )
        release_cdf = (
            np.searchsorted(release_values, all_values[:-1], side="right")
            / release_values.size
        )
        distance = float(np.sum(np.abs(original_cdf - release_cdf) * deltas))
    if normalization_scale is not None:
        distance /= normalization_scale
    return max(0.0, distance)


def calculate_utility_discrepancy(
    original: pd.DataFrame,
    release: pd.DataFrame,
    config: UtilityMetricConfig,
) -> UtilityResult:
    """Calculate weighted normalized discrepancy between ``RT_0`` and ``RT``."""

    feature_distances: dict[str, float] = {}
    weighted_contributions: dict[str, float] = {}
    normalization_scales: dict[str, float | None] = {}

    for feature in config.features:
        if feature.name not in original.columns:
            raise KeyError(f"original table is missing feature {feature.name!r}")
        if feature.name not in release.columns:
            raise KeyError(f"release table is missing feature {feature.name!r}")
        original_column = cast(Iterable[object], original[feature.name])
        release_column = cast(Iterable[object], release[feature.name])

        if feature.feature_type is FeatureType.CATEGORICAL:
            distance = jensen_shannon_divergence(
                original_column,
                release_column,
                smoothing=feature.smoothing,
                log_base=feature.log_base,
                missing_policy=feature.missing_policy,
                suppressed_values=feature.suppressed_values,
            )
            scale: float | None = None
        else:
            original_numeric = _ordered_values(original_column, feature)
            release_numeric = _ordered_values(release_column, feature)
            scale = _normalization_scale(original_numeric, release_numeric, feature)
            distance = wasserstein_distance(
                original_numeric,
                release_numeric,
                normalization_scale=scale,
            )
        feature_distances[feature.name] = distance
        weighted_contributions[feature.name] = feature.weight * distance
        normalization_scales[feature.name] = scale

    aggregate = math.fsum(weighted_contributions.values())
    passes = (
        None
        if config.tau_max is None
        else validate_utility_threshold(aggregate, config.tau_max)
    )
    return UtilityResult(
        distance=aggregate,
        feature_distances=feature_distances,
        weighted_contributions=weighted_contributions,
        normalization_scales=normalization_scales,
        tau_max=config.tau_max,
        passes_threshold=passes,
    )


def validate_utility_threshold(distance: float, tau_max: float) -> bool:
    """Validate an aggregate discrepancy and return whether it is releasable."""

    _require_finite_nonnegative(distance, "distance")
    _require_finite_nonnegative(tau_max, "tau_max")
    return distance <= tau_max


def _categorical_values(
    values: Iterable[object],
    *,
    missing_policy: MissingValuePolicy,
    suppressed_values: frozenset[str],
) -> list[Hashable]:
    result: list[Hashable] = []
    for value in values:
        if _is_missing(value, suppressed_values):
            if missing_policy is MissingValuePolicy.ERROR:
                raise ValueError("feature contains a missing or suppressed value")
            if missing_policy is MissingValuePolicy.EXCLUDE:
                continue
            result.append(cast(Hashable, _MISSING_CATEGORY))
            continue
        if not isinstance(value, Hashable):
            raise TypeError("categorical values must be hashable scalars")
        result.append(_category_key(value))
    return result


def _probabilities(
    counts: Counter[Hashable],
    categories: tuple[Hashable, ...],
    *,
    smoothing: float,
) -> npt.NDArray[np.float64]:
    denominator = math.fsum(counts.values()) + smoothing * len(categories)
    return np.asarray(
        [
            (counts.get(category, 0) + smoothing) / denominator
            for category in categories
        ],
        dtype=np.float64,
    )


def _kl_divergence(
    probabilities: npt.NDArray[np.float64],
    midpoint: npt.NDArray[np.float64],
    log_base: float,
) -> float:
    nonzero = probabilities > 0.0
    terms = probabilities[nonzero] * (
        np.log(probabilities[nonzero] / midpoint[nonzero]) / math.log(log_base)
    )
    return float(np.sum(terms))


def _ordered_values(
    values: Iterable[object], feature: UtilityFeatureConfig
) -> list[float]:
    missing_policy = feature.missing_policy
    if missing_policy is MissingValuePolicy.AUTO:
        missing_policy = MissingValuePolicy.EXCLUDE
    suppression_set = frozenset(feature.suppressed_values)
    order = {
        _category_key(value): float(index)
        for index, value in enumerate(feature.ordinal_order)
    }
    result: list[float] = []
    for value in values:
        if _is_missing(value, suppression_set):
            if missing_policy is MissingValuePolicy.ERROR:
                raise ValueError(
                    f"feature {feature.name!r} contains a missing or suppressed value"
                )
            continue
        if order:
            key = _category_key(value)
            if key not in order:
                raise ValueError(
                    f"feature {feature.name!r} contains value outside ordinal_order: {value!r}"
                )
            result.append(order[key])
        else:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"feature {feature.name!r} must contain numeric values")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"feature {feature.name!r} must contain finite values")
            result.append(numeric)
    if not result:
        raise ValueError(f"feature {feature.name!r} has no usable values")
    return result


def _normalization_scale(
    original: list[float],
    release: list[float],
    feature: UtilityFeatureConfig,
) -> float | None:
    strategy = feature.normalization
    if strategy is NormalizationStrategy.AUTO:
        strategy = (
            NormalizationStrategy.FIXED
            if feature.feature_type is FeatureType.ORDINAL and feature.ordinal_order
            else NormalizationStrategy.COMBINED_RANGE
        )
        if strategy is NormalizationStrategy.FIXED:
            return float(len(feature.ordinal_order) - 1)
    if strategy is NormalizationStrategy.NONE:
        return None
    if strategy is NormalizationStrategy.FIXED:
        return feature.ground_metric_scale
    values = (
        original
        if strategy is NormalizationStrategy.REFERENCE_RANGE
        else original + release
    )
    scale = max(values) - min(values)
    if scale == 0.0:
        if min(original + release) == max(original + release):
            # The distributions are identical point masses, hence W1 is zero.
            return 1.0
        raise ValueError(
            f"feature {feature.name!r} has zero reference range; "
            "use combined_range or a fixed ground_metric_scale"
        )
    return scale


def _finite_float_array(values: Iterable[float], label: str) -> npt.NDArray[np.float64]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{label} distribution must be a non-empty one-dimensional sample"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{label} distribution must contain only finite values")
    array.sort()
    return array


def _is_missing(value: object, suppressed_values: frozenset[str]) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return value in suppressed_values
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, np.floating):
        return math.isnan(cast(float, value))
    return False


def _category_key(value: object) -> Hashable:
    if not isinstance(value, Hashable):
        raise TypeError("category and ordinal values must be hashable scalars")
    return (type(value), value)


def _require_finite_nonnegative(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")


def _require_finite_positive(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
