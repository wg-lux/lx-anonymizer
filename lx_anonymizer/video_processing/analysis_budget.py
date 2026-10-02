"""Fail closed when exhaustive analysis cannot meet its bounded time budget."""

import json
import logging
import math
from collections import deque

logger = logging.getLogger(__name__)
DEFAULT_MAX_ANALYSIS_SECONDS = 3600.0
PROJECTION_MIN_FRAMES = 32
PROJECTION_MIN_SECONDS = 30.0


class FrameAnalysisBudgetExceeded(TimeoutError):
    """The invocation cannot complete exhaustive coverage within its budget."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Exhaustive frame analysis time budget: {reason}")


def metadata_probe_frames(total_frames: int, maximum_probes: int = 128) -> set[int]:
    """Spread probes through interval subdivision, without assuming monotonic text.

    The decoder consumes these positions in source order. These probes cannot
    prove absence of an overlay in an uninspected frame.
    """
    if total_frames <= 0 or maximum_probes < 2:
        raise ValueError("Metadata probes require frames and at least two probes")
    probes = {0, total_frames - 1}
    intervals = deque([(0, total_frames - 1)])
    while intervals and len(probes) < maximum_probes:
        start, end = intervals.popleft()
        middle = (start + end) // 2
        if middle in (start, end):
            continue
        probes.add(middle)
        intervals.extend(((start, middle), (middle, end)))
    return probes


def validate_analysis_budget(maximum_seconds: float) -> None:
    if not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("Frame analysis time budget must be finite and positive")


def check_analysis_budget(
    *,
    total_frames: int,
    completed_frames: int,
    elapsed_seconds: float,
    maximum_seconds: float,
    project_completion: bool = True,
) -> None:
    """Check elapsed time and measured whole-analysis throughput, never skip frames.

    The estimate intentionally includes decoding and metadata extraction. It is
    an admission guard, not a claim of future throughput or clinical recall.
    """
    validate_analysis_budget(maximum_seconds)
    if (
        total_frames <= 0
        or not 0 <= completed_frames <= total_frames
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("Invalid frame analysis progress")

    projected_seconds = (
        elapsed_seconds * (total_frames / completed_frames)
        if completed_frames
        else None
    )
    reason: str | None = None
    if elapsed_seconds >= maximum_seconds:
        reason = "deadline_exceeded"
    elif (
        project_completion
        and completed_frames >= PROJECTION_MIN_FRAMES
        and elapsed_seconds >= PROJECTION_MIN_SECONDS
        and projected_seconds is not None
        and projected_seconds > maximum_seconds
    ):
        reason = "projected_deadline_exceeded"
    if reason is None:
        return

    logger.warning(
        json.dumps(
            {
                "event": "frame_analysis.budget_exceeded",
                "reason": reason,
                "total_frames": total_frames,
                "completed_frames": completed_frames,
                "elapsed_seconds": elapsed_seconds,
                "maximum_seconds": maximum_seconds,
                "projected_seconds": projected_seconds,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    raise FrameAnalysisBudgetExceeded(reason)
