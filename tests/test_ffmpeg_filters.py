import pytest

from lx_anonymizer.video_processing.ffmpeg_filters import (
    build_frame_drop_filters,
    frame_ranges,
)


def test_frame_ranges_merges_contiguous_indices() -> None:
    assert frame_ranges([3, 1, 2, 8, 8, 10]) == [(1, 3), (8, 8), (10, 10)]


def test_frame_drop_filters_use_time_ranges_for_audio() -> None:
    filters = build_frame_drop_filters([10, 11, 12], 25.0)

    assert "eq(n\\,10)" in filters.video_filter
    assert "eq(n\\,11)" in filters.video_filter
    assert "eq(n\\,12)" in filters.video_filter
    assert "between(t\\,0.4\\,0.52)" in filters.audio_filter


def test_frame_drop_filters_compress_large_video_conditions() -> None:
    filters = build_frame_drop_filters(range(70), 30.0)

    assert "between(n\\,0\\,69)" in filters.video_filter
    assert "between(t\\,0\\,2.333333333)" in filters.audio_filter


def test_frame_drop_filters_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="frame indices"):
        build_frame_drop_filters([-1], 30.0)

    with pytest.raises(ValueError, match="frame_rate"):
        build_frame_drop_filters([1], 0.0)
