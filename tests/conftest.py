# tests/conftest.py
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_central_video_format():
    """
    Centrally mocks detect_video_format to always return a valid 1080p canvas.
    Keeps tests running on dummy/fake video file names independent from ffprobe.
    """
    target = "lx_anonymizer.video_processing.video_utils.detect_video_format"
    with patch(target) as mock_probe:
        mock_probe.return_value = {
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "width": 1920,
            "height": 1080,
            "has_audio": True,
            "container": "mov,mp4,m4a,3gp,3g2,mj2",
            "can_stream_copy": True,
        }
        yield mock_probe
