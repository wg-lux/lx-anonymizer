import subprocess
from pathlib import Path
from typing import cast

import pytest

from lx_anonymizer.video_processing import video_processor as video_processor_module
from lx_anonymizer.video_processing.video_encoder import VideoEncoder
from lx_anonymizer.video_processing.video_processor import VideoProcessor


class _FakeEncoder:
    calls: list[tuple[str, bool]]

    def __init__(self) -> None:
        self.calls = []

    def build_encoder_cmd(
        self, quality_mode: str = "balanced", fallback: bool = False
    ) -> list[str]:
        self.calls.append((quality_mode, fallback))
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-profile:v",
            "high",
        ]


def test_mask_video_uses_configured_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    encoder = _FakeEncoder()
    processor = VideoProcessor(cast(VideoEncoder, encoder))
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"source")
    captured_cmd: list[str] = []

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is True
        captured_cmd.extend(cmd)
        output_video.write_bytes(b"encoded")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(video_processor_module.subprocess, "run", fake_run)

    ok = processor.mask_video(
        input_video,
        {
            "x": 10,
            "y": 20,
            "width": 100,
            "height": 80,
            "image_width": 200,
            "image_height": 160,
        },
        output_video,
    )

    assert ok is True
    assert encoder.calls == [("balanced", False)]
    assert captured_cmd[:6] == [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(input_video),
        "-vf",
    ]
    video_codec_index = captured_cmd.index("-c:v")
    audio_codec_index = captured_cmd.index("-c:a")
    assert captured_cmd[video_codec_index : video_codec_index + 2] == [
        "-c:v",
        "libx264",
    ]
    assert video_codec_index < audio_codec_index
    assert captured_cmd[audio_codec_index : audio_codec_index + 2] == [
        "-c:a",
        "copy",
    ]
