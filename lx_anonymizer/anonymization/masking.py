import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from lx_anonymizer.video_processing.video_encoder import VideoEncoder
from lx_anonymizer.video_processing import video_utils

logger = logging.getLogger(__name__)


class MaskApplication:
    def __init__(
        self, preferred_encoder: Dict[str, Any], device_name: str = "olympus_cv_1500"
    ):
        self.preferred_encoder = preferred_encoder
        self.video_encoder = VideoEncoder()
        self.build_encoder_cmd = self.video_encoder.build_encoder_cmd

        # Default mask configuration based on olympus_cv_1500_mask.json
        self.default_mask_config = {
            "image_width": 1920,
            "image_height": 1080,
            "endoscope_image_x": 550,
            "endoscope_image_y": 0,
            "endoscope_image_width": 1350,
            "endoscope_image_height": 1080,
        }
        self.device_name = device_name

    def mask_video_streaming(
        self,
        input_video: Path,
        mask_config: Dict[str, Any],
        output_video: Path,
        use_named_pipe: bool = False,
    ) -> bool:
        """
        Apply video masking using streaming approach to crop sensitive areas while preserving endoscope image.

        Based on olympus_cv_1500_mask.json:
        - Endoscope image: x=550-1900, y=0-1080 (preserve this area)
        - Sensitive areas: x=0-550 (exclude this area by cropping)

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
            for key, value in mask_config.items():
                if value is not None:
                    effective_config[key] = value

            def _coerce_int(value: Any, fallback: int) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return fallback

            endoscope_x = _coerce_int(effective_config.get("endoscope_image_x"), 550)
            endoscope_y = _coerce_int(effective_config.get("endoscope_image_y"), 0)
            endoscope_w = _coerce_int(
                effective_config.get("endoscope_image_width"), 1350
            )
            endoscope_h = _coerce_int(
                effective_config.get("endoscope_image_height"), 1080
            )
            image_width = _coerce_int(effective_config.get("image_width"), 0)
            image_height = _coerce_int(effective_config.get("image_height"), 0)

            if image_width <= 0 or image_height <= 0:
                format_info = video_utils.detect_video_format(input_video)
                image_width = image_width or format_info.get("width", 0)
                image_height = image_height or format_info.get("height", 0)

            if endoscope_w <= 0 or endoscope_h <= 0:
                logger.error(
                    "Invalid crop size: width=%d height=%d", endoscope_w, endoscope_h
                )
                return False

            crop_x = max(0, endoscope_x)
            crop_y = max(0, endoscope_y)
            crop_w = endoscope_w
            crop_h = endoscope_h

            if image_width > 0:
                if crop_x >= image_width:
                    logger.error(
                        "Crop x=%d exceeds image width=%d", crop_x, image_width
                    )
                    return False
                max_crop_w = image_width - crop_x
                if crop_w > max_crop_w:
                    logger.warning(
                        "Crop width %d exceeds image width; clamping to %d",
                        crop_w,
                        max_crop_w,
                    )
                    crop_w = max_crop_w

            if image_height > 0:
                if crop_y >= image_height:
                    logger.error(
                        "Crop y=%d exceeds image height=%d", crop_y, image_height
                    )
                    return False
                max_crop_h = image_height - crop_y
                if crop_h > max_crop_h:
                    logger.warning(
                        "Crop height %d exceeds image height; clamping to %d",
                        crop_h,
                        max_crop_h,
                    )
                    crop_h = max_crop_h

            if crop_w <= 0 or crop_h <= 0:
                logger.error(
                    "Crop area collapsed: x=%d y=%d w=%d h=%d",
                    crop_x,
                    crop_y,
                    crop_w,
                    crop_h,
                )
                return False

            if (
                image_width > 0
                and image_height > 0
                and crop_x == 0
                and crop_y == 0
                and crop_w == image_width
                and crop_h == image_height
            ):
                logger.warning("No cropping needed - endoscope covers entire frame")
                import shutil

                shutil.copy2(input_video, output_video)
                return True

            if crop_x % 2 != 0:
                crop_x += 1
                crop_w -= 1
                logger.debug(
                    "Adjusted crop x to even boundary: x=%d w=%d", crop_x, crop_w
                )
            if crop_y % 2 != 0:
                crop_y += 1
                crop_h -= 1
                logger.debug(
                    "Adjusted crop y to even boundary: y=%d h=%d", crop_y, crop_h
                )
            if crop_w % 2 != 0:
                crop_w -= 1
                logger.debug("Adjusted crop width to even: w=%d", crop_w)
            if crop_h % 2 != 0:
                crop_h -= 1
                logger.debug("Adjusted crop height to even: h=%d", crop_h)

            if crop_w <= 0 or crop_h <= 0:
                logger.error(
                    "Crop area invalid after alignment: x=%d y=%d w=%d h=%d",
                    crop_x,
                    crop_y,
                    crop_w,
                    crop_h,
                )
                return False

            vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
            encoder_args = self.build_encoder_cmd("balanced")

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
                "Cropping to endoscope image region x=%d y=%d w=%d h=%d",
                crop_x,
                crop_y,
                crop_w,
                crop_h,
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
                output_info = video_utils.detect_video_format(output_video)
                out_w = output_info.get("width", 0)
                out_h = output_info.get("height", 0)
                if out_w and out_h and (out_w != crop_w or out_h != crop_h):
                    logger.warning(
                        "Masked video dimensions %dx%d do not match expected crop %dx%d",
                        out_w,
                        out_h,
                        crop_w,
                        crop_h,
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

    def _load_mask(self) -> Dict[str, Any]:
        masks_dir = Path(__file__).parent / "masks"
        mask_file = masks_dir / f"{self.device_name}_mask.json"
        stub = {
            "image_width": 1920,
            "image_height": 1080,
            "x": 550,
            "y": 0,
            "width": 1350,
            "height": 1080,
        }

        try:
            with mask_file.open() as f:
                return json.load(f)  # works if file is valid
        except (FileNotFoundError, json.JSONDecodeError):
            # create or overwrite with a fresh stub
            masks_dir.mkdir(parents=True, exist_ok=True)
            with mask_file.open("w") as f:
                json.dump(stub, f, indent=2)
            logger.warning(
                "Created or repaired mask file %s – please verify coordinates.",
                mask_file,
            )
            return stub

        except (json.JSONDecodeError, IOError) as e:
            logger.error(
                f"Failed to load/create mask configuration for {self.device_name}: {e}"
            )
            raise FileNotFoundError(
                f"Could not load or create mask configuration for {self.device_name}: {e}"
            )
