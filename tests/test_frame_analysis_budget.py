"""AAA admission tests using measured progress without waiting for real time."""

import math

import pytest
from pydantic import ValidationError

from lx_anonymizer.processing_contracts import MetadataAnalysisCoverage
from lx_anonymizer.video_processing.analysis_budget import (
    FrameAnalysisBudgetExceeded,
    check_analysis_budget,
    metadata_probe_frames,
    validate_analysis_budget,
)


def test_gc02_observed_throughput_rejects_run_after_warmup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: 105307 source frames, observed 11574 calls / 34263.91 seconds.
    elapsed_seconds = 32 * (34263.91067412098 / 11574)

    # Act / Assert: about 95 seconds exposes an infeasible scan, not nine hours.
    with pytest.raises(FrameAnalysisBudgetExceeded, match="projected_deadline"):
        check_analysis_budget(
            total_frames=105307,
            completed_frames=32,
            elapsed_seconds=elapsed_seconds,
            maximum_seconds=3600,
        )
    assert elapsed_seconds < 100
    assert '"event": "frame_analysis.budget_exceeded"' in caplog.text
    assert '"completed_frames": 32' in caplog.text


@pytest.mark.parametrize("completed,elapsed", [(1, 40), (31, 90), (32, 29)])
def test_projection_waits_for_both_warmup_thresholds(
    completed: int, elapsed: float
) -> None:
    # Arrange / Act: initialization and short bursts are not stable throughput.
    check_analysis_budget(
        total_frames=105307,
        completed_frames=completed,
        elapsed_seconds=elapsed,
        maximum_seconds=3600,
    )
    # Assert: no rejection before the warmup completes.


@pytest.mark.parametrize("completed", [0, 1, 32])
def test_deadline_applies_even_without_completed_warmup(completed: int) -> None:
    # Arrange / Act / Assert
    with pytest.raises(FrameAnalysisBudgetExceeded, match="deadline_exceeded"):
        check_analysis_budget(
            total_frames=100,
            completed_frames=completed,
            elapsed_seconds=3600,
            maximum_seconds=3600,
        )


def test_short_scan_and_explicitly_sized_budget_can_complete() -> None:
    # Arrange / Act / Assert: slow OCR can be feasible for a short source.
    check_analysis_budget(
        total_frames=61,
        completed_frames=32,
        elapsed_seconds=100,
        maximum_seconds=3600,
    )
    # A caller can explicitly authorize a larger budget; no silent clamping.
    check_analysis_budget(
        total_frames=105307,
        completed_frames=32,
        elapsed_seconds=100,
        maximum_seconds=400000,
    )


@pytest.mark.parametrize("maximum", [0, -1, math.inf, math.nan])
def test_invalid_budget_fails_at_configuration_boundary(maximum: float) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="finite and positive"):
        validate_analysis_budget(maximum)


def test_interval_probes_cover_both_ends_and_middle_without_claiming_exhaustion() -> (
    None
):
    # Arrange / Act
    probes = metadata_probe_frames(105307, maximum_probes=128)
    # Assert
    assert len(probes) == 128
    assert {0, 52653, 105306} <= probes
    assert all(0 <= frame < 105307 for frame in probes)


@pytest.mark.parametrize("total_frames", [1, 2, 3, 61])
def test_probe_schedule_can_cover_short_sources(total_frames: int) -> None:
    # Arrange / Act / Assert
    assert metadata_probe_frames(total_frames) == set(range(total_frames))


@pytest.mark.parametrize(
    "frames,complete", [([0, 0], False), ([3], False), ([0, 1, 2], False), ([0], True)]
)
def test_coverage_cannot_claim_inspection_it_does_not_have(
    frames: list[int],
    complete: bool,
) -> None:
    # Arrange / Act / Assert: partial metadata is usable only with honest coverage.
    with pytest.raises(ValidationError):
        MetadataAnalysisCoverage(
            total_frames=3,
            inspected_frame_numbers=frames,
            complete=complete,
            mode="interval_sampled",
            cap_reason="deadline_exceeded",
            elapsed_seconds=1.0,
            maximum_seconds=1.0,
        )
