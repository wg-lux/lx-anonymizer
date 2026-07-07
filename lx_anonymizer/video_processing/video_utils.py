import json
import math
import os
import subprocess
import tempfile
import logging
from fractions import Fraction
from pathlib import Path
from typing import cast
from contextlib import contextmanager

from lx_dtypes.models.contracts.video_format import VideoFormatInfo

logger = logging.getLogger(__name__)

DEFAULT_FFPROBE_TIMEOUT_SECONDS = 10.0
DEFAULT_VIDEO_FRAME_RATE = 30.0

UNKNOWN_VIDEO_FORMAT = VideoFormatInfo()


def can_use_stream_copy(
    video_stream: dict[str, object], audio_streams: list[dict[str, object]]
) -> bool:
    """Determine if FFmpeg -c copy is viable based on codecs and pixel formats."""
    good_video_codecs = {"h264", "h265", "hevc", "vp8", "vp9", "av1"}
    good_audio_codecs = {"aac", "mp3", "opus", "vorbis"}

    video_codec = str(video_stream.get("codec_name", "")).lower()
    if video_codec not in good_video_codecs:
        return False

    for audio_stream in audio_streams:
        if str(audio_stream.get("codec_name", "")).lower() not in good_audio_codecs:
            return False

    pixel_format = str(video_stream.get("pix_fmt", ""))
    # High bit-depth or 4:2:2 chroma often requires re-encoding for compatibility
    if any(tag in pixel_format for tag in ["10le", "422", "12le"]):
        return False

    return True


def detect_video_format(
    video_path: Path,
    *,
    timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Analyze video format using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-nostdin",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds,
        )
        info = cast(dict[str, object], json.loads(result.stdout))
        info_map = _coerce_mapping(info)

        streams = _coerce_stream_list(info_map.get("streams", []))
        v_stream = _find_first_stream(streams, "video")
        a_streams = _find_streams(streams, "audio")
        format_info = _coerce_mapping(info_map.get("format", {}))

        probe = VideoFormatInfo.model_validate(
            {
                "video_codec": v_stream.get("codec_name", "unknown"),
                "pixel_format": v_stream.get("pix_fmt", "unknown"),
                "width": v_stream.get("width", 0),
                "height": v_stream.get("height", 0),
                "has_audio": len(a_streams) > 0,
                "container": format_info.get("format_name", "unknown"),
                "can_stream_copy": can_use_stream_copy(v_stream, a_streams),
            }
        )
        return probe.model_dump()
    except (
        json.JSONDecodeError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.warning(f"Metadata probe failed for {video_path}: {e}")
        return UNKNOWN_VIDEO_FORMAT.model_dump()


def detect_video_frame_rate(
    video_path: Path,
    *,
    timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
    default_frame_rate: float = DEFAULT_VIDEO_FRAME_RATE,
) -> float:
    """Return the first video stream's frame rate, falling back to 30 fps."""
    fallback = _positive_finite_float(default_frame_rate) or DEFAULT_VIDEO_FRAME_RATE
    try:
        cmd = [
            "ffprobe",
            "-nostdin",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-print_format",
            "json",
            str(video_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds,
        )
        info = cast(dict[str, object], json.loads(result.stdout))
        streams = _coerce_stream_list(_coerce_mapping(info).get("streams", []))
        if not streams:
            return fallback

        video_stream = streams[0]
        for key in ("avg_frame_rate", "r_frame_rate"):
            frame_rate = parse_frame_rate(video_stream.get(key))
            if frame_rate is not None:
                return frame_rate
        return fallback
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.warning(f"Frame-rate probe failed for {video_path}: {e}")
        return fallback


def parse_frame_rate(value: object) -> float | None:
    """Parse ffprobe frame-rate values such as '30000/1001' or '30'."""
    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return _positive_finite_float(float(value))

    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped or stripped == "0/0" or stripped.upper() == "N/A":
        return None

    try:
        return _positive_finite_float(float(Fraction(stripped)))
    except (ValueError, ZeroDivisionError):
        return None


@contextmanager
def named_pipe(suffix: str = ".mp4"):
    """Creates a temporary FIFO for in-memory streaming."""
    temp_dir = Path(tempfile.mkdtemp(prefix="anony_pipe_"))
    pipe_path = temp_dir / f"stream{suffix}"
    os.mkfifo(pipe_path)
    try:
        yield pipe_path
    finally:
        if pipe_path.exists():
            pipe_path.unlink()
        temp_dir.rmdir()


def _coerce_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    raw_mapping = cast(dict[object, object], value)
    return {str(key): raw_value for key, raw_value in raw_mapping.items()}


def _coerce_stream_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    streams: list[dict[str, object]] = []
    for item in cast(list[object], value):
        if isinstance(item, dict):
            raw_item = cast(dict[object, object], item)
            streams.append(_coerce_mapping(raw_item))
    return streams


def _find_first_stream(
    streams: list[dict[str, object]], codec_type: str
) -> dict[str, object]:
    for stream in streams:
        if str(stream.get("codec_type", "")) == codec_type:
            return stream
    return {}


def _find_streams(
    streams: list[dict[str, object]], codec_type: str
) -> list[dict[str, object]]:
    return [
        stream for stream in streams if str(stream.get("codec_type", "")) == codec_type
    ]


def _positive_finite_float(value: float) -> float | None:
    if math.isfinite(value) and value > 0:
        return value
    return None
