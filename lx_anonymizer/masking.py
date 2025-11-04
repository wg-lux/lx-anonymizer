import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from .video_encoder import VideoEncoder

logger = logging.getLogger(__name__)


class MaskApplication:
    def __init__(self, preferred_encoder: Dict[str, Any]):
        self.preferred_encoder = preferred_encoder
        self.video_encoder = VideoEncoder(
            mask_video_streaming=False, create_mask_config_from_roi=False
        )
        self._build_encoder_cmd = self.video_encoder._build_encoder_cmd

        # Default mask configuration based on olympus_cv_1500_mask.json
        self.default_mask_config = {
            "image_width": 1920,
            "image_height": 1080,
            "endoscope_image_x": 550,
            "endoscope_image_y": 0,
            "endoscope_image_width": 1350,
            "endoscope_image_height": 1080,
        }

    def mask_video_streaming(
        self,
        input_video: Path,
        mask_config: Dict[str, Any],
        output_video: Path,
        use_named_pipe: bool = False,
    ) -> bool:
        """
        Apply video masking using streaming approach to mask sensitive areas while preserving endoscope image.

        Based on olympus_cv_1500_mask.json:
        - Endoscope image: x=550-1900, y=0-1080 (preserve this area)
        - Sensitive areas: x=0-550 (mask this area with black)

        Args:
            input_video: Path to input video file
            mask_config: Dictionary containing mask coordinates
            output_video: Path for output masked video
            use_named_pipe: Whether to use named pipes for streaming (currently not implemented)

        Returns:
            True if masking succeeded, False otherwise
        """
        # Named pipe functionality would be implemented here in the future
        if use_named_pipe:
            logger.debug("Named pipe functionality requested but not yet implemented")

        try:
            # Use default config if not provided or merge with defaults
            effective_config = self.default_mask_config.copy()
            effective_config.update(mask_config)

            endoscope_x = effective_config.get("endoscope_image_x", 550)
            endoscope_y = effective_config.get("endoscope_image_y", 0)
            endoscope_w = effective_config.get("endoscope_image_width", 1350)
            endoscope_h = effective_config.get("endoscope_image_height", 1080)
            image_width = effective_config.get("image_width", 1920)
            image_height = effective_config.get("image_height", 1080)

            # Always use drawbox filters to mask sensitive areas (not crop)
            mask_filters = []

            # Mask left side (sensitive area) if endoscope doesn't start at x=0
            if endoscope_x > 0:
                mask_filters.append(
                    f"drawbox=0:0:{endoscope_x}:{image_height}:color=black@1:t=fill"
                )
                logger.debug(
                    "Masking left side sensitive area: x=0 to x=%d", endoscope_x
                )

            # Mask right side if endoscope doesn't extend to full width
            right_start = endoscope_x + endoscope_w
            if right_start < image_width:
                right_width = image_width - right_start
                mask_filters.append(
                    f"drawbox={right_start}:0:{right_width}:{image_height}:color=black@1:t=fill"
                )
                logger.debug(
                    "Masking right side area: x=%d to x=%d", right_start, image_width
                )

            # Mask top area if endoscope doesn't start at y=0
            if endoscope_y > 0:
                mask_filters.append(
                    f"drawbox={endoscope_x}:0:{endoscope_w}:{endoscope_y}:color=black@1:t=fill"
                )
                logger.debug("Masking top area: y=0 to y=%d", endoscope_y)

            # Mask bottom area if endoscope doesn't extend to full height
            bottom_start = endoscope_y + endoscope_h
            if bottom_start < image_height:
                bottom_height = image_height - bottom_start
                mask_filters.append(
                    f"drawbox={endoscope_x}:{bottom_start}:{endoscope_w}:{bottom_height}:color=black@1:t=fill"
                )
                logger.debug(
                    "Masking bottom area: y=%d to y=%d", bottom_start, image_height
                )

            if not mask_filters:
                logger.warning("No masking needed - endoscope covers entire frame")
                # Just copy the video without changes
                import shutil

                shutil.copy2(input_video, output_video)
                return True

            vf = ",".join(mask_filters)
            encoder_args = self._build_encoder_cmd("balanced")

            cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(input_video),
                "-vf",
                vf,
                *encoder_args,
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output_video),
            ]

            logger.info(
                "Masking sensitive areas with %d regions, preserving endoscope image at x=%d-y=%d",
                len(mask_filters),
                endoscope_x,
                endoscope_y,
            )
            logger.debug("FFmpeg command: %s", " ".join(cmd))

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug("Masking output: %s", result.stderr)

            # Verify output
            if output_video.exists() and output_video.stat().st_size > 0:
                # Compare file sizes to ensure reasonable output
                input_size = input_video.stat().st_size
                output_size = output_video.stat().st_size
                size_ratio = output_size / input_size if input_size > 0 else 0

                if size_ratio < 0.1:  # Output suspiciously small
                    logger.warning(
                        "Output video much smaller than input (%.1f%%)",
                        size_ratio * 100,
                    )

                logger.info(
                    "Successfully created masked video: %s (size ratio: %.1f%%)",
                    output_video,
                    size_ratio * 100,
                )
                return True
            else:
                logger.error("Masked video is empty or missing")
                return False

        except subprocess.CalledProcessError as e:
            logger.error("Streaming mask failed: %s", e.stderr)
            return False
        except (OSError, IOError) as e:
            logger.error("File operation failed during masking: %s", e)
            return False

    def create_mask_config_from_roi(
        self, endoscope_image_roi: dict[str, int]
    ) -> Dict[str, Any]:
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
