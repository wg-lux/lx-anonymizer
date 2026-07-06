import subprocess
import logging
import tempfile
from pathlib import Path

from lx_dtypes.models.contracts.video_processing import (
    VideoEncoderConfig,
    VideoMaskConfig,
)

logger = logging.getLogger(__name__)


class VideoProcessingService:
    """
    Encapsulates FFmpeg operations for masking, frame removal,
    pixel format conversion, and stream copying.
    """

    def __init__(self, preferred_encoder: dict[str, object]):
        self.preferred_encoder = VideoEncoderConfig.model_validate(preferred_encoder)

    # ----------------------------
    # Stream Copy & Conversion
    # ----------------------------
    def stream_copy_video(self, input_video: Path, output_video: Path) -> bool:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(input_video),
            "-c",
            "copy",
            str(output_video),
        ]
        logger.info(f"Stream copy video: {input_video} -> {output_video}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_video.exists() and output_video.stat().st_size > 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Stream copy failed: {e.stderr}")
            return False

    # ----------------------------
    # Mask Overlay
    # ----------------------------
    def mask_video(
        self, input_video: Path, mask_config: dict[str, object], output_video: Path
    ) -> bool:
        """
        Apply black mask outside ROI region.
        """
        try:
            mask = VideoMaskConfig.model_validate(mask_config)
            x, y = mask.x, mask.y
            w, h = mask.width, mask.height
            iw, ih = mask.image_width, mask.image_height
            draw_boxes: list[str] = []
            # left, right, top, bottom masks
            if x > 0:
                draw_boxes.append(f"drawbox=0:0:{x}:{ih}:color=black@1:t=fill")
            if x + w < iw:
                draw_boxes.append(
                    f"drawbox={x + w}:0:{iw - (x + w)}:{ih}:color=black@1:t=fill"
                )
            if y > 0:
                draw_boxes.append(f"drawbox={x}:0:{w}:{y}:color=black@1:t=fill")
            if y + h < ih:
                draw_boxes.append(
                    f"drawbox={x}:{y + h}:{w}:{ih - (y + h)}:color=black@1:t=fill"
                )

            vf = ",".join(draw_boxes)
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(input_video),
                "-vf",
                vf,
                "-c:a",
                "copy",
                str(output_video),
            ]
            logger.debug(f"Masking ffmpeg command: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_video.exists() and output_video.stat().st_size > 0
        except Exception as e:
            logger.error(f"Masking failed: {e}")
            return False

    def frame_removal(
        self, input_video: Path, frames_to_remove: list[int], output_video: Path
    ) -> bool:
        """
        Remove specified frames from the video.
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # Extract frames
                extract_cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_video),
                    str(tmpdir_path / "frame_%05d.png"),
                ]
                subprocess.run(extract_cmd, capture_output=True, text=True, check=True)

                # Remove specified frames
                for frame_num in frames_to_remove:
                    frame_file = tmpdir_path / f"frame_{frame_num:05d}.png"
                    if frame_file.exists():
                        frame_file.unlink()

                # Reassemble video
                reassemble_cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-framerate",
                    "30",
                    "-i",
                    str(tmpdir_path / "frame_%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_video),
                ]
                subprocess.run(
                    reassemble_cmd, capture_output=True, text=True, check=True
                )

            return output_video.exists() and output_video.stat().st_size > 0
        except Exception as e:
            logger.error(f"Frame removal failed: {e}")
            return False
