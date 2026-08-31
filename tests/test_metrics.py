import math

import numpy as np
import pandas as pd
import pytest

from lx_anonymizer.metrics.utility import (
    FeatureType,
    MissingValuePolicy,
    NormalizationStrategy,
    UtilityFeatureConfig,
    UtilityMetricConfig,
    calculate_utility_discrepancy,
    jensen_shannon_divergence,
    validate_utility_threshold,
    wasserstein_distance,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (["a", "a", "b"], ["a", "b", "b"]),
        (["a"], ["b"]),
        (["a", None, "b"], ["a", "*", "b"]),
    ],
)
def test_jsd_is_nonnegative_and_symmetric(
    left: list[str | None], right: list[str | None]
) -> None:
    forward = jensen_shannon_divergence(left, right)
    reverse = jensen_shannon_divergence(right, left)

    assert forward >= 0.0
    assert math.isclose(forward, reverse)


def test_jsd_identity_and_zero_frequency_categories() -> None:
    values = ["a", "a", "b", "c"]

    assert math.isclose(jensen_shannon_divergence(values, values), 0.0, abs_tol=1e-12)
    assert math.isclose(jensen_shannon_divergence(["a"], ["b"]), 1.0)
    assert math.isfinite(jensen_shannon_divergence(["a"], ["b"], smoothing=0.5))


def test_missing_and_suppressed_categories_share_one_explicit_bucket() -> None:
    assert (
        jensen_shannon_divergence(
            ["a", None, np.nan],
            ["a", "*", "REDACTED"],
            suppressed_values=("*", "REDACTED"),
        )
        == 0.0
    )
    assert (
        jensen_shannon_divergence(
            ["a", None],
            ["a", "*"],
            missing_policy=MissingValuePolicy.EXCLUDE,
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="missing or suppressed"):
        jensen_shannon_divergence(
            ["a", "*"], ["a", "b"], missing_policy=MissingValuePolicy.ERROR
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [([0.0, 1.0, 2.0], [0.0, 1.0, 3.0]), ([0.0], [2.0])],
)
def test_wasserstein_is_nonnegative_symmetric_and_normalizable(
    left: list[float], right: list[float]
) -> None:
    forward = wasserstein_distance(left, right)

    assert forward >= 0.0
    assert math.isclose(forward, wasserstein_distance(right, left))
    assert math.isclose(
        wasserstein_distance(left, right, normalization_scale=2.0), forward / 2.0
    )


def test_wasserstein_identity_of_indiscernibles() -> None:
    values = [1.0, 3.0, 7.0, 9.0]
    assert wasserstein_distance(values, values) == 0.0


def test_dataframe_aggregate_weighting_ordinal_mapping_and_threshold() -> None:
    original = pd.DataFrame(
        {
            "diagnosis": ["a", "a", "b", "b"],
            "age": [10.0, 20.0, 30.0, 40.0],
            "severity": ["low", "medium", "high", "high"],
        }
    )
    release = pd.DataFrame(
        {
            "diagnosis": ["a", "b", "b", "b"],
            "age": [20.0, 30.0, 40.0, 50.0],
            "severity": ["medium", "medium", "high", "high"],
        }
    )
    config = UtilityMetricConfig(
        features=(
            UtilityFeatureConfig("diagnosis", FeatureType.CATEGORICAL, 0.2),
            UtilityFeatureConfig("age", FeatureType.CONTINUOUS, 0.5),
            UtilityFeatureConfig(
                "severity",
                FeatureType.ORDINAL,
                0.3,
                ordinal_order=("low", "medium", "high"),
            ),
        ),
        tau_max=0.25,
    )

    result = calculate_utility_discrepancy(original, release, config)

    expected = math.fsum(
        feature.weight * result.feature_distances[feature.name]
        for feature in config.features
    )
    assert math.isclose(result.distance, expected)
    assert result.normalization_scales["age"] == 40.0
    assert result.normalization_scales["severity"] == 2.0
    assert result.passes_threshold is (result.distance <= 0.25)
    assert result.within_threshold == result.passes_threshold
    assert result.d_util == result.distance


def test_dataframe_numeric_suppression_is_excluded_by_auto_policy() -> None:
    original = pd.DataFrame({"age": [0.0, 1.0, 2.0, 3.0]})
    release = pd.DataFrame({"age": [0.0, "*", 2.0, np.nan]})
    config = UtilityMetricConfig(
        features=(
            UtilityFeatureConfig(
                "age",
                FeatureType.CONTINUOUS,
                1.0,
                normalization=NormalizationStrategy.FIXED,
                ground_metric_scale=3.0,
            ),
        )
    )

    result = calculate_utility_discrepancy(original, release, config)

    assert math.isclose(result.distance, 1.0 / 6.0)
    assert result.passes_threshold is None


@pytest.mark.parametrize("tau_max", [0.0, 0.5, 1.0])
def test_threshold_validation_is_inclusive(tau_max: float) -> None:
    assert validate_utility_threshold(tau_max, tau_max)
    assert not validate_utility_threshold(tau_max + 0.1, tau_max)


def test_configuration_rejects_invalid_weights_and_thresholds() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        UtilityMetricConfig(
            features=(
                UtilityFeatureConfig("a", FeatureType.CATEGORICAL, 0.4),
                UtilityFeatureConfig("b", FeatureType.CONTINUOUS, 0.4),
            )
        )
    with pytest.raises(ValueError, match="tau_max"):
        UtilityMetricConfig(
            features=(UtilityFeatureConfig("a", FeatureType.CATEGORICAL, 1.0),),
            tau_max=-0.1,
        )


def test_missing_feature_and_unknown_ordinal_value_fail_loudly() -> None:
    ordinal_config = UtilityMetricConfig(
        features=(
            UtilityFeatureConfig(
                "severity",
                FeatureType.ORDINAL,
                1.0,
                ordinal_order=("low", "high"),
            ),
        )
    )
    with pytest.raises(KeyError, match="missing feature"):
        calculate_utility_discrepancy(
            pd.DataFrame({"other": [1]}), pd.DataFrame({"other": [1]}), ordinal_config
        )
    with pytest.raises(ValueError, match="outside ordinal_order"):
        calculate_utility_discrepancy(
            pd.DataFrame({"severity": ["low"]}),
            pd.DataFrame({"severity": ["medium"]}),
            ordinal_config,
        )
