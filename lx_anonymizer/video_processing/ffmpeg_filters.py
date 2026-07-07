from dataclasses import dataclass
import math
from collections.abc import Sequence

DEFAULT_FRAME_RATE = 30.0


@dataclass(frozen=True)
class FrameDropFilters:
    video_filter: str
    audio_filter: str


def build_frame_drop_filters(
    frames_to_remove: Sequence[int],
    frame_rate: float,
) -> FrameDropFilters:
    """
    Build FFmpeg filters that remove video frames by frame index and matching
    audio spans by timestamp.
    """
    clean = _validated_frame_indices(frames_to_remove)
    fps = validated_frame_rate(frame_rate)

    if not clean:
        return FrameDropFilters(
            video_filter="select='1',setpts=N/FRAME_RATE/TB",
            audio_filter="aselect='1',asetpts=N/SR/TB",
        )

    video_condition = _frame_index_condition(clean)
    audio_condition = _frame_time_condition(frame_ranges(clean), fps)
    return FrameDropFilters(
        video_filter=f"select='not({video_condition})',setpts=N/FRAME_RATE/TB",
        audio_filter=f"aselect='not({audio_condition})',asetpts=N/SR/TB",
    )


def frame_ranges(indices: Sequence[int]) -> list[tuple[int, int]]:
    clean = _validated_frame_indices(indices)
    if not clean:
        return []

    ranges: list[tuple[int, int]] = []
    start = clean[0]
    end = clean[0]
    for value in clean[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append((start, end))
        start = value
        end = value
    ranges.append((start, end))
    return ranges


def validated_frame_rate(frame_rate: float) -> float:
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(
            f"frame_rate must be a positive finite value, got {frame_rate}"
        )
    return frame_rate


def _validated_frame_indices(indices: Sequence[int]) -> list[int]:
    clean = sorted(set(indices))
    invalid = [idx for idx in clean if idx < 0]
    if invalid:
        raise ValueError(f"frame indices must be non-negative, got {invalid}")
    return clean


def _frame_index_condition(indices: Sequence[int]) -> str:
    if len(indices) <= 64:
        return "+".join(f"eq(n\\,{idx})" for idx in indices)

    terms: list[str] = []
    for start, end in frame_ranges(indices):
        if start == end:
            terms.append(f"eq(n\\,{start})")
        else:
            terms.append(f"between(n\\,{start}\\,{end})")
    return "+".join(terms)


def _frame_time_condition(ranges: Sequence[tuple[int, int]], frame_rate: float) -> str:
    terms: list[str] = []
    for start_frame, end_frame in ranges:
        start_seconds = _format_seconds(start_frame / frame_rate)
        end_seconds = _format_seconds((end_frame + 1) / frame_rate)
        terms.append(f"between(t\\,{start_seconds}\\,{end_seconds})")
    return "+".join(terms)


def _format_seconds(value: float) -> str:
    formatted = f"{value:.9f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"
