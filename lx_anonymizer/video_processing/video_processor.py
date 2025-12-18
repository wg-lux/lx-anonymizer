import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
from lx_anonymizer.video_processing import video_utils
from lx_anonymizer.video_processing.video_encoder import VideoEncoder

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Encapsulates FFmpeg operations for masking, frame removal,
    pixel format conversion, and stream copying.
    """

    def __init__(self, encoder_component: VideoEncoder):
        self.encoder = encoder_component

    def stream_copy_video(
        self,
        input_video: Path,
        output_video: Path,
        format_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determines the fastest way to save the video (Copy vs Fast Re-encode)."""
        if format_info is None:
            format_info = video_utils.detect_video_format(input_video)

        try:
            if format_info["can_stream_copy"]:
                # The fastest possible path: no transcoding
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_video),
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(output_video),
                ]
                logger.info("Performing pure stream copy (ultra-fast)")
            else:
                pixel_fmt = format_info.get("pixel_format", "unknown")
                # Handle specific high-depth formats that break most players
                if any(x in pixel_fmt for x in ["10le", "422"]):
                    return self._stream_copy_with_pixel_conversion(
                        input_video, output_video
                    )

                # Default: Fast hardware-accelerated re-encode
                encoder_args = self.encoder.build_encoder_cmd("fast")
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_video),
                    *encoder_args,
                    "-c:a",
                    "copy",
                    str(output_video),
                ]
                logger.info(
                    f"Re-encoding video using {self.encoder.preferred_encoder['type']}"
                )

            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_video.exists()
        except Exception as e:
            logger.error(f"Failed to copy/convert video: {e}")
            return False

    # ----------------------------
    # Stream Copy & Conversion
    # ----------------------------
    def _pixel_conversion_fallback(
        self, input_video: Path, output_video: Path, target_pixel_format: str
    ) -> bool:
        """
        Fallback pixel format conversion using CPU encoding.

        Args:
            input_video: Source video path
            output_video: Destination video path
            target_pixel_format: Target pixel format

        Returns:
            True if conversion succeeded
        """
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(input_video),
                "-vf",
                f"format={target_pixel_format}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-c:a",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(output_video),
            ]

            logger.info(
                f"CPU fallback pixel conversion: {input_video} -> {output_video}"
            )
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            return output_video.exists() and output_video.stat().st_size > 0

        except subprocess.CalledProcessError as e:
            logger.error(f"CPU fallback pixel conversion failed: {e.stderr}")
            return False

    def _stream_copy_with_pixel_conversion(
        self,
        input_video: Path,
        output_video: Path,
        target_pixel_format: str = "yuv420p",
    ) -> bool:
        """Fast conversion of pixel format only, keeping audio via copy."""
        try:
            # Use the encoder command builder from the video_encoder component
            encoder_args = self.encoder.build_encoder_cmd("balanced")

            cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(input_video),
                "-vf",
                f"format={target_pixel_format}",
                *encoder_args,
                "-c:a",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(output_video),
            ]

            logger.info(
                f"Normalizing pixel format using {self.encoder.preferred_encoder['type']}"
            )
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_video.exists() and output_video.stat().st_size > 0

        except subprocess.CalledProcessError:
            if self.encoder.preferred_encoder["type"] == "nvenc":
                logger.warning("NVENC conversion failed, falling back to CPU.")
                return self._pixel_conversion_fallback(
                    input_video, output_video, target_pixel_format
                )
            return False

    # ----------------------------
    # Mask Overlay
    # ----------------------------
    def mask_video(
        self, input_video: Path, mask_config: Dict[str, Any], output_video: Path
    ) -> bool:
        """
        Apply black mask outside ROI region.
        """
        try:
            x, y = mask_config["x"], mask_config["y"]
            w, h = mask_config["width"], mask_config["height"]
            iw, ih = (
                mask_config.get("image_width", 1920),
                mask_config.get("image_height", 1080),
            )
            draw_boxes = []
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
        self, input_video: Path, frames_to_remove: List[int], output_video: Path
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

    def extract_frames(
        self, video_path: Path, output_dir: Path, fps: int = 1
    ) -> List[Path]:
        """Dumps high-quality PNGs for OCR processing."""
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "1",
            "-pix_fmt",
            "rgb24",
            str(output_dir / "frame_%04d.png"),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return sorted(output_dir.glob("frame_*.png"))

    def remove_frames_streaming(
        self, input_video: Path, frames: List[int], output_video: Path
    ) -> bool:
        """Removes specific frames using a select filter and named pipes."""
        if not frames:
            return self.stream_copy_video(input_video, output_video)

        idx_filter = "+".join([f"eq(n\\,{idx})" for idx in frames])
        vf = f"select='not({idx_filter})',setpts=N/FRAME_RATE/TB"
        af = f"aselect='not({idx_filter})',asetpts=N/SR/TB"

        with video_utils.named_pipe() as pipe:
            # Command 1: Filter to Pipe
            filter_cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(input_video),
                "-vf",
                vf,
                "-af",
                af,
                "-f",
                "matroska",
                str(pipe),
            ]

            # Command 2: Pipe to Final Output (using Centralized Encoder)
            encoder_args = self.encoder.build_encoder_cmd("balanced")
            copy_cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(pipe),
                *encoder_args,
                "-c:a",
                "copy",
                str(output_video),
            ]

            proc = subprocess.Popen(filter_cmd, stderr=subprocess.PIPE)
            subprocess.run(copy_cmd, check=True)
            proc.communicate()

        return output_video.exists()
