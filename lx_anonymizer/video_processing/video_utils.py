import json
import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def can_use_stream_copy(video_stream: Dict, audio_streams: List[Dict]) -> bool:
    """Determine if FFmpeg -c copy is viable based on codecs and pixel formats."""
    good_video_codecs = {"h264", "h265", "hevc", "vp8", "vp9", "av1"}
    good_audio_codecs = {"aac", "mp3", "opus", "vorbis"}

    video_codec = video_stream.get("codec_name", "").lower()
    if video_codec not in good_video_codecs:
        return False

    for audio_stream in audio_streams:
        if audio_stream.get("codec_name", "").lower() not in good_audio_codecs:
            return False

    pixel_format = video_stream.get("pix_fmt", "")
    # High bit-depth or 4:2:2 chroma often requires re-encoding for compatibility
    if any(tag in pixel_format for tag in ["10le", "422", "12le"]):
        return False

    return True


def detect_video_format(video_path: Path) -> Dict[str, Any]:
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        v_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
        a_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]

        return {
            "video_codec": v_stream.get("codec_name", "unknown"),
            "pixel_format": v_stream.get("pix_fmt", "unknown"),
            "width": int(v_stream.get("width", 0)),
            "height": int(v_stream.get("height", 0)),
            "has_audio": len(a_streams) > 0,
            "container": info["format"].get("format_name", "unknown"),
            "can_stream_copy": can_use_stream_copy(v_stream, a_streams),
        }
    except Exception as e:
        logger.warning(f"Metadata probe failed for {video_path}: {e}")
        return {"can_stream_copy": False, "has_audio": True}


@contextmanager
def named_pipe(suffix=".mp4"):
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
