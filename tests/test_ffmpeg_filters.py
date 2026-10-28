import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from lx_anonymizer.video_processing import video_utils
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


@pytest.mark.parametrize("frames", [[], [10, 11, 12], list(range(70))])
def test_frame_drop_filters_never_retime_retained_frames(frames: list[int]) -> None:
    filters = build_frame_drop_filters(frames, 30.0)

    assert "setpts" not in filters.video_filter
    assert "asetpts" not in filters.audio_filter


def test_frame_removal_keeps_recorded_timestamps_and_gap(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are not available")
    source = tmp_path / "source.mp4"
    output = tmp_path / "removed.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=25",
            "-frames:v",
            "20",
            "-c:v",
            "libx264",
            "-video_track_timescale",
            "600",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    filters = build_frame_drop_filters([5, 6, 7], 25.0)
    # Timestamp-preserving flags as supplied by the endoreg-db execution hooks.
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-copyts",
            "-i",
            str(source),
            "-vf",
            filters.video_filter,
            "-an",
            "-c:v",
            "libx264",
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "demux",
            "-video_track_timescale",
            "600",
            str(output),
        ],
        check=True,
        capture_output=True,
    )

    def ticks(path: Path) -> list[int]:
        probed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return [int(value.strip(",")) for value in probed.stdout.split()]

    source_ticks = ticks(source)
    assert ticks(output) == [
        tick for index, tick in enumerate(source_ticks) if index not in {5, 6, 7}
    ]


@pytest.mark.parametrize(
    ("probe_output", "expected"),
    [("1/600\n", Fraction(1, 600)), ("1/90000\n", Fraction(1, 90_000))],
)
def test_detect_video_time_base_reads_stream_time_base(
    monkeypatch: pytest.MonkeyPatch, probe_output: str, expected: Fraction
) -> None:
    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=probe_output, stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    assert video_utils.detect_video_time_base(Path("in.mp4")) == expected
    assert video_utils.timestamp_preserving_output_args(Path("in.mp4"))[-1] == str(
        expected.denominator
    )


@pytest.mark.parametrize(
    ("probe_output", "message"),
    [("", "unavailable"), ("N/A\n", "unavailable"), ("0/1\n", "positive")],
)
def test_detect_video_time_base_fails_without_guessing(
    monkeypatch: pytest.MonkeyPatch, probe_output: str, message: str
) -> None:
    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=probe_output, stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match=message):
        video_utils.detect_video_time_base(Path("in.mp4"))


@pytest.mark.parametrize("probe_output", ["1001/30000\n", "N/A\n", ""])
def test_unusable_time_base_keeps_removal_working_without_pinned_timescale(
    monkeypatch: pytest.MonkeyPatch, probe_output: str
) -> None:
    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=probe_output, stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    assert video_utils.timestamp_preserving_output_args(Path("in.mp4")) == [
        "-fps_mode",
        "passthrough",
        "-enc_time_base",
        "demux",
    ]


def test_failed_time_base_probe_keeps_removal_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_run(
        cmd: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(video_utils.subprocess, "run", failing_run)

    assert "-video_track_timescale" not in video_utils.timestamp_preserving_output_args(
        Path("in.mp4")
    )
