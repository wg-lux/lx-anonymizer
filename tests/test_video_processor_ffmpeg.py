import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest

from lx_anonymizer.video_processing import video_processor as processor_module
from lx_anonymizer.video_processing.video_encoder import VideoEncoder
from lx_anonymizer.video_processing.video_processor import VideoProcessor


class StubVideoEncoder:
    preferred_encoder: dict[str, object] = {"type": "cpu"}

    def build_encoder_cmd(
        self,
        quality_mode: str = "balanced",
        fallback: bool = False,
    ) -> list[str]:
        _ = quality_mode
        _ = fallback
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


def _video_processor() -> VideoProcessor:
    return VideoProcessor(cast(VideoEncoder, StubVideoEncoder()))


def test_mask_video_uses_configured_video_encoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processor = _video_processor()
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"input")
    captured_cmd: list[str] = []

    def fake_run(
        cmd: list[str],
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        _ = args
        _ = kwargs
        captured_cmd[:] = cmd
        Path(cmd[-1]).write_bytes(b"output")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(processor_module.subprocess, "run", fake_run)

    ok = processor.mask_video(
        input_video,
        {"x": 10, "y": 20, "width": 100, "height": 80},
        output_video,
    )

    assert ok is True
    assert "-vf" in captured_cmd
    assert captured_cmd[captured_cmd.index("-c:v") + 1] == "libx264"
    assert "-movflags" in captured_cmd


def test_extract_frames_uses_fast_png_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processor = _video_processor()
    output_dir = tmp_path / "frames"
    captured_cmd: list[str] = []

    def fake_run(
        cmd: list[str],
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        _ = args
        _ = kwargs
        captured_cmd[:] = cmd
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.png").write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(processor_module.subprocess, "run", fake_run)

    frames = processor.extract_frames(tmp_path / "input.mp4", output_dir, fps=2)

    assert frames == [output_dir / "frame_0001.png"]
    assert "-compression_level" in captured_cmd
    assert captured_cmd[captured_cmd.index("-compression_level") + 1] == "0"
    assert "-q:v" not in captured_cmd


def test_remove_frames_streaming_uses_single_process_and_removes_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processor = _video_processor()
    output_video = tmp_path / "output.mp4"
    captured_cmd: list[str] = []

    def fake_frame_rate(video_path: Path) -> float:
        _ = video_path
        return 30.0

    def fake_format(video_path: Path) -> dict[str, object]:
        _ = video_path
        return {"has_audio": True}

    def fake_run(
        cmd: list[str],
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        _ = args
        _ = kwargs
        captured_cmd[:] = cmd
        Path(cmd[-1]).write_bytes(b"output")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fail_popen(*args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs
        raise AssertionError("remove_frames_streaming should not start a pipe process")

    monkeypatch.setattr(
        processor_module.video_utils, "detect_video_frame_rate", fake_frame_rate
    )

    def fake_time_base(video_path: Path) -> Fraction:
        _ = video_path
        return Fraction(1, 600)

    monkeypatch.setattr(
        processor_module.video_utils, "detect_video_format", fake_format
    )
    monkeypatch.setattr(
        processor_module.video_utils, "detect_video_time_base", fake_time_base
    )
    monkeypatch.setattr(processor_module.subprocess, "run", fake_run)
    monkeypatch.setattr(processor_module.subprocess, "Popen", fail_popen)

    ok = processor.remove_frames_streaming(
        tmp_path / "input.mp4",
        [10, 11, 12],
        output_video,
    )

    assert ok is True
    assert "-f" not in captured_cmd
    assert "matroska" not in captured_cmd
    assert "-vf" in captured_cmd
    assert "-af" not in captured_cmd
    assert "-an" in captured_cmd

    video_filter = captured_cmd[captured_cmd.index("-vf") + 1]
    assert "eq(n\\,10)" in video_filter
    assert "setpts" not in video_filter
    assert captured_cmd.index("-copyts") < captured_cmd.index("-i")
    assert captured_cmd[captured_cmd.index("-fps_mode") + 1] == "passthrough"
    assert captured_cmd[captured_cmd.index("-enc_time_base") + 1] == "demux"
    assert captured_cmd[captured_cmd.index("-video_track_timescale") + 1] == "600"
    assert captured_cmd[captured_cmd.index("-c:v") + 1] == "libx264"


def _probe_ticks(ffprobe: str, path: Path) -> list[int]:
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


@pytest.mark.parametrize("timescale", [600, 1000, 90_000])
def test_real_frame_removal_keeps_recorded_timestamps_without_host_hooks(
    tmp_path: Path, timescale: int
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are not available")
    source = tmp_path / "source.mp4"
    output = tmp_path / "removed.mp4"
    # A 30 then 24 frames-per-second clip yields variable spacing; the offset
    # reproduces a nonzero recording origin.
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=30",
            "-vf",
            "setpts='if(lt(N,10),N/30,10/30+(N-10)/24)/TB+7/TB'",
            "-frames:v",
            "20",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-video_track_timescale",
            str(timescale),
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    assert _video_processor().remove_frames_streaming(source, [4, 5, 12], output)

    source_ticks = _probe_ticks(ffprobe, source)
    assert len(source_ticks) == 20
    assert _probe_ticks(ffprobe, output) == [
        tick for index, tick in enumerate(source_ticks) if index not in {4, 5, 12}
    ]
