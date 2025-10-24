import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from .video_encoder import VideoEncoder

logger = logging.getLogger(__name__)


class MaskApplication:
    def __init__(self, preferred_encoder: Dict[str, Any]):
        self.preferred_encoder = preferred_encoder
        self.video_encoder = VideoEncoder(mask_video_streaming=False, create_mask_config_from_roi=False)
        self._build_encoder_cmd = self.video_encoder._build_encoder_cmd

    def mask_video_streaming(self, input_video: Path, mask_config: Dict[str, Any], output_video: Path, use_named_pipe: bool = True) -> bool:
        """
        Apply video masking using streaming approach with optional named pipes.

        Args:
            input_video: Path to input video file
            mask_config: Dictionary containing mask coordinates
            output_video: Path for output masked video
            use_named_pipe: Whether to use named pipes for streaming

        Returns:
            True if masking succeeded, False otherwise
        """
        try:
            endoscope_x = mask_config.get("endoscope_image_x", 0)
            endoscope_y = mask_config.get("endoscope_image_y", 0)
            endoscope_w = mask_config.get("endoscope_image_width", 640)
            endoscope_h = mask_config.get("endoscope_image_height", 480)

            # Check if we can use simple crop (most efficient)
            if endoscope_y == 0 and endoscope_h == mask_config.get("image_height", 1080):
                # Simple crop - use single-pass processing for maximum efficiency
                crop_filter = f"crop=in_w-{endoscope_x}:in_h:{endoscope_x}:0"
                encoder_args = self._build_encoder_cmd("balanced")

                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_video),
                    "-vf",
                    crop_filter,
                    *encoder_args,
                    "-c:a",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output_video),
                ]

                logger.info(f"Direct crop masking (single pass) using {self.preferred_encoder['type']}: {crop_filter}")
                logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.debug(f"Direct masking output: {result.stderr}")

            else:
                # Complex masking - use drawbox filters
                mask_filters = []

                # Build mask rectangles (same logic as before)
                if endoscope_x > 0:
                    mask_filters.append(f"drawbox=0:0:{endoscope_x}:{mask_config.get('image_height', 1080)}:color=black@1:t=fill")

                right_start = endoscope_x + endoscope_w
                image_width = mask_config.get("image_width", 1920)
                if right_start < image_width:
                    right_width = image_width - right_start
                    mask_filters.append(f"drawbox={right_start}:0:{right_width}:{mask_config.get('image_height', 1080)}:color=black@1:t=fill")

                if endoscope_y > 0:
                    mask_filters.append(f"drawbox={endoscope_x}:0:{endoscope_w}:{endoscope_y}:color=black@1:t=fill")

                bottom_start = endoscope_y + endoscope_h
                image_height = mask_config.get("image_height", 1080)
                if bottom_start < image_height:
                    bottom_height = image_height - bottom_start
                    mask_filters.append(f"drawbox={endoscope_x}:{bottom_start}:{endoscope_w}:{bottom_height}:color=black@1:t=fill")

                vf = ",".join(mask_filters)

                # Use hardware-optimized encoding for complex masks
                encoder_args = self._build_encoder_cmd("fast")
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_video),
                    "-vf",
                    vf,
                    *encoder_args,  # Use hardware-optimized encoder
                    "-c:a",
                    "copy",  # Always copy audio
                    "-movflags",
                    "+faststart",
                    str(output_video),
                ]

                logger.info(f"Complex mask processing with {len(mask_filters)} regions using {self.preferred_encoder['type']}")
                logger.debug(f"FFmpeg command: {' '.join(cmd)}")

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.debug(f"Complex masking output: {result.stderr}")

            # Verify output
            if output_video.exists() and output_video.stat().st_size > 0:
                # Compare file sizes to ensure reasonable output
                input_size = input_video.stat().st_size
                output_size = output_video.stat().st_size
                size_ratio = output_size / input_size if input_size > 0 else 0

                if size_ratio < 0.1:  # Output suspiciously small
                    logger.warning(f"Output video much smaller than input ({size_ratio:.1%})")

                logger.info(f"Successfully created masked video: {output_video} (size ratio: {size_ratio:.1%})")
                return True
            else:
                logger.error("Masked video is empty or missing")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Streaming mask failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Streaming mask error: {e}")
            return False

    def create_mask_config_from_roi(self, endoscope_image_roi: dict[str, int]) -> Dict[str, Any]:
        """
        Create mask config dictionary from ROI.
        Args:
            endoscope_image_roi: ROI dictionary
        Returns:
            Mask config dictionary
        """
        # Beispiel: Übernehme die Werte direkt
        return {
            "endoscope_image_x": endoscope_image_roi.get("x"),
            "endoscope_image_y": endoscope_image_roi.get("y"),
            "endoscope_image_width": endoscope_image_roi.get("width"),
            "endoscope_image_height": endoscope_image_roi.get("height"),
            "image_width": endoscope_image_roi.get("image_width"),
            "image_height": endoscope_image_roi.get("image_height"),
        }
