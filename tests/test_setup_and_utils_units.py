import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, cast

import numpy as np
import pytest
import subprocess

from lx_anonymizer.region_processing import box_operations as box_ops
from lx_anonymizer.setup import directory_setup
from lx_anonymizer.video_processing import video_utils

_REAL_DETECT_VIDEO_FORMAT = video_utils.detect_video_format


@pytest.fixture(autouse=True)
def use_real_video_format(
    monkeypatch: pytest.MonkeyPatch,
    mock_central_video_format: object,
) -> Iterator[None]:
    """
    Let the central autouse mock start, then restore the real probe for this module.
    """
    monkeypatch.setattr(video_utils, "detect_video_format", _REAL_DETECT_VIDEO_FORMAT)
    yield


def test_str_to_path_and_create_directories(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"

    assert directory_setup._str_to_path(str(a)) == a  # pyright: ignore[reportPrivateUsage]
    assert directory_setup._str_to_path(b) == b  # pyright: ignore[reportPrivateUsage]

    created = directory_setup.create_directories([a, b])
    assert created == [a, b]
    assert a.exists()
    assert b.exists()


def test_directory_helpers_create_and_reuse_paths(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    temp_root = tmp_path / "temp_root"

    got_main = directory_setup.create_main_directory(main_dir)
    assert got_main == main_dir
    assert main_dir.exists()

    got_results = directory_setup.create_results_directory(main_dir)
    assert got_results == main_dir / "results"
    assert got_results.exists()

    got_models = directory_setup.create_model_directory(main_dir)
    assert got_models == main_dir / "models"
    assert got_models.exists()

    temp_dir, base_dir, csv_dir = directory_setup.create_temp_directory(
        default_temp_directory=temp_root,
        default_main_directory=main_dir,
    )
    assert temp_dir == temp_root / "temp"
    assert base_dir == main_dir
    assert csv_dir == main_dir / "csv_training_data"
    assert temp_dir.exists()
    assert csv_dir.exists()

    blur_dir = directory_setup.create_blur_directory(main_dir)
    assert blur_dir == main_dir / "blurred_results"
    assert blur_dir.exists()

    # Re-running should take existing paths and return the same values.
    assert directory_setup.create_results_directory(main_dir) == got_results
    assert directory_setup.create_model_directory(main_dir) == got_models
    assert directory_setup.create_blur_directory(main_dir) == blur_dir


def test_can_use_stream_copy_decisions() -> None:
    assert video_utils.can_use_stream_copy(
        {"codec_name": "h264", "pix_fmt": "yuv420p"},
        [{"codec_name": "aac"}],
    )
    assert not video_utils.can_use_stream_copy(
        {"codec_name": "mpeg2video", "pix_fmt": "yuv420p"},
        [{"codec_name": "aac"}],
    )
    assert not video_utils.can_use_stream_copy(
        {"codec_name": "h264", "pix_fmt": "yuv420p10le"},
        [{"codec_name": "aac"}],
    )
    assert not video_utils.can_use_stream_copy(
        {"codec_name": "h264", "pix_fmt": "yuv420p"},
        [{"codec_name": "pcm_s16le"}],
    )


def test_detect_video_format_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }

    captured_cmd: list[str] = []
    captured_kwargs: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        captured_cmd.extend(cast(list[str], args[0]))
        captured_kwargs.update(kwargs)
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)
    info = video_utils.detect_video_format(Path("x.mp4"))
    assert "-nostdin" not in captured_cmd
    assert captured_kwargs["timeout"] == video_utils.DEFAULT_FFPROBE_TIMEOUT_SECONDS
    assert captured_kwargs["stdin"] == subprocess.DEVNULL
    assert info == {
        "video_codec": "h264",
        "pixel_format": "yuv420p",
        "width": 1920,
        "height": 1080,
        "has_audio": True,
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "can_stream_copy": True,
    }

    def fake_run_fail(*args: object, **kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(returncode=1, cmd="ffprobe")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run_fail)
    fallback = video_utils.detect_video_format(Path("x.mp4"))
    assert fallback == {
        "video_codec": "unknown",
        "pixel_format": "unknown",
        "width": 0,
        "height": 0,
        "has_audio": True,
        "container": "unknown",
        "can_stream_copy": False,
    }

    def fake_run_timeout(*args: object, **kwargs: object) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=10.0)

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run_timeout)
    timed_out = video_utils.detect_video_format(Path("x.mp4"))
    assert timed_out == {
        "video_codec": "unknown",
        "pixel_format": "unknown",
        "width": 0,
        "height": 0,
        "has_audio": True,
        "container": "unknown",
        "can_stream_copy": False,
    }

    def fake_run_missing_keys(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=json.dumps({}))

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run_missing_keys)
    missing_keys = video_utils.detect_video_format(Path("x.mp4"))
    assert missing_keys["width"] == 0
    assert missing_keys["height"] == 0
    assert missing_keys["container"] == "unknown"
    assert missing_keys["can_stream_copy"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30000/1001", 29.97002997002997),
        ("25", 25.0),
        (24, 24.0),
        (23.976, 23.976),
    ],
)
def test_parse_frame_rate_accepts_ffprobe_values(
    value: object, expected: float
) -> None:
    frame_rate = video_utils.parse_frame_rate(value)
    assert frame_rate is not None
    assert math.isclose(frame_rate, expected)


@pytest.mark.parametrize("value", ["0/0", "N/A", "", None, False, -1, 0])
def test_parse_frame_rate_rejects_invalid_values(value: object) -> None:
    assert video_utils.parse_frame_rate(value) is None


def test_detect_video_frame_rate_success_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmd: list[str] = []
    captured_kwargs: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        captured_cmd.extend(cast(list[str], args[0]))
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            stdout=json.dumps({"streams": [{"avg_frame_rate": "30000/1001"}]})
        )

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)
    frame_rate = video_utils.detect_video_frame_rate(Path("x.mp4"))

    assert math.isclose(frame_rate, 29.97002997002997)
    assert "-nostdin" not in captured_cmd
    assert captured_kwargs["timeout"] == video_utils.DEFAULT_FFPROBE_TIMEOUT_SECONDS
    assert captured_kwargs["stdin"] == subprocess.DEVNULL

    def fake_run_no_streams(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=json.dumps({"streams": []}))

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run_no_streams)
    assert math.isclose(
        video_utils.detect_video_frame_rate(Path("x.mp4"), default_frame_rate=24.0),
        24.0,
    )

    def fake_run_fail(*args: object, **kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(returncode=1, cmd="ffprobe")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run_fail)
    assert math.isclose(
        video_utils.detect_video_frame_rate(Path("x.mp4"), default_frame_rate=0.0),
        video_utils.DEFAULT_VIDEO_FRAME_RATE,
    )


def test_named_pipe_context_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temp_root = tmp_path / "pipes"
    temp_root.mkdir()
    made = {"i": 0}

    def fake_mkdtemp(prefix: str) -> str:
        made["i"] += 1
        p = temp_root / f"{prefix}{made['i']}"
        p.mkdir()
        return str(p)

    def fake_mkfifo(path: Path) -> None:
        path.touch()

    monkeypatch.setattr(video_utils.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(video_utils.os, "mkfifo", fake_mkfifo)

    with video_utils.named_pipe(".mkv") as pipe_path:
        assert pipe_path.name == "stream.mkv"
        assert pipe_path.exists()
        parent = pipe_path.parent

    assert not pipe_path.exists()
    assert not parent.exists()


def test_box_operations_core_behaviors(monkeypatch: pytest.MonkeyPatch) -> None:
    text_with_boxes = [
        ("A", (0, 0, 5, 5)),
        ("BB", (6, 0, 10, 5)),
        ("CC", (0, 20, 5, 25)),
    ]
    filtered = box_ops.filter_empty_boxes(text_with_boxes, min_text_len=2)
    assert filtered == [("BB", (6, 0, 10, 5)), ("CC", (0, 20, 5, 25))]

    combined = box_ops.combine_boxes(
        [
            ("left", (0, 10, 10, 20)),
            ("right", (15, 10, 25, 20)),
            ("next", (0, 30, 5, 35)),
        ]
    )
    assert combined == [
        ("left right", (0, 10, 25, 20)),
        ("next", (0, 30, 5, 35)),
    ]
    combined_with_y_tolerance = box_ops.combine_boxes(
        [("one", (0, 10, 10, 20)), ("two", (15, 14, 25, 24))]
    )
    assert combined_with_y_tolerance == [("one two", (0, 10, 25, 24))]

    assert box_ops.close_to_box((10, 10, 20, 20), (15, 15, 30, 30))
    assert not box_ops.close_to_box((10, 10, 20, 20), (30, 30, 40, 40))

    assert box_ops.make_box_from_device_list(1, 2, 3, 4) == (1, 2, 4, 6)

    image = np.full((20, 20, 3), 10, dtype=np.uint8)
    assert box_ops.get_dominant_color(image) == (10, 10, 10)
    assert box_ops.get_dominant_color(image, (100, 100, 101, 101)) == (255, 255, 255)

    new_box = box_ops.find_or_create_close_box(
        phrase_box=(10, 10, 20, 20),
        boxes=[(50, 12, 60, 22), (80, 12, 90, 22)],
        image_width=120,
        min_offset=20,
    )
    assert new_box == (50, 12, 60, 22)

    created_box = box_ops.find_or_create_close_box(
        phrase_box=(90, 10, 100, 20),
        boxes=[],
        image_width=110,
        min_offset=20,
    )
    assert created_box == (100, 10, 110, 20)

    image2 = np.zeros((100, 100, 3), dtype=np.uint8)

    def fake_dominant(
        _img: np.ndarray, box: tuple[int, int, int, int] | None = None
    ) -> tuple[int, int, int]:
        color_map = {
            (20, 20, 40, 40): (0, 0, 0),
            (20, 10, 40, 20): (100, 0, 0),  # upper differs
            (20, 40, 40, 50): (100, 0, 0),  # lower differs
            (10, 10, 20, 50): (100, 0, 0),  # left differs after y-extension
            (40, 10, 50, 50): (0, 0, 0),  # right same
        }
        assert box is not None
        return color_map[box]

    monkeypatch.setattr(box_ops, "get_dominant_color", fake_dominant)
    extended = box_ops.extend_boxes_if_needed(
        image2,
        [(20, 20, 40, 40)],
        extension_margin=10,
        color_threshold=5,
    )
    assert extended == [(10, 10, 40, 50)]
