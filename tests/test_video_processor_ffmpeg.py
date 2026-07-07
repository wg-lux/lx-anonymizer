import subprocess
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


def test_remove_frames_streaming_uses_single_process_time_based_audio_filter(
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
    monkeypatch.setattr(
        processor_module.video_utils, "detect_video_format", fake_format
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
    assert "-af" in captured_cmd

    video_filter = captured_cmd[captured_cmd.index("-vf") + 1]
    audio_filter = captured_cmd[captured_cmd.index("-af") + 1]
    assert "eq(n\\,10)" in video_filter
    assert "between(t\\,0.333333333\\,0.433333333)" in audio_filter
    assert "eq(n" not in audio_filter
    assert captured_cmd[captured_cmd.index("-c:v") + 1] == "libx264"
    assert captured_cmd[captured_cmd.index("-c:a") + 1] == "aac"
