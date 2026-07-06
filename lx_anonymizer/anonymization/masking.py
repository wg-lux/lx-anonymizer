import json
import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, cast

from lx_dtypes.models.contracts.video_processing import (
    VideoMaskConfig,
    VideoMaskRegionCore,
)
from lx_anonymizer.video_processing.video_encoder import VideoEncoder
from lx_anonymizer.video_processing import video_utils

logger = logging.getLogger(__name__)


class MaskMode(str, Enum):
    PRESERVE_DIMENSIONS = "preserve_dimensions"
    CROP = "crop"


@dataclass(frozen=True)
class DimensionBackfillResult:
    status: str
    source_dimensions: tuple[int, int]
    anonymized_dimensions: tuple[int, int]
    repaired: bool = False
    detail: str = ""


class MaskApplication:
    def __init__(
        self,
        preferred_encoder: Mapping[str, object],
        device_name: str = "olympus_cv_1500",
    ):
        self.preferred_encoder = dict(preferred_encoder)
        self.video_encoder = VideoEncoder()
        self.build_encoder_cmd = self.video_encoder.build_encoder_cmd
        self.device_name = device_name

        # Default mask configuration based on olympus_cv_1500_mask.json
        self.default_mask_config: VideoMaskConfig = self._load_mask()

    def mask_video_streaming(
        self,
        input_video: Path,
        mask_config: Mapping[str, object],
        output_video: Path,
        use_named_pipe: bool = False,
        mode: MaskMode | str = MaskMode.PRESERVE_DIMENSIONS,
    ) -> bool:
        """
        Apply video masking using a streaming FFmpeg command.

        The default mode preserves input dimensions and masks everything outside
        the endoscope ROI with black boxes. The crop mode is retained only for
        explicit compatibility with older outputs.

        Args:
            input_video: Path to input video file
            mask_config: Dictionary containing mask coordinates (endoscope_image_* or x/y/width/height)
            output_video: Path for output masked video
            use_named_pipe: Whether to use named pipes for streaming (currently not implemented)
            mode: preserve_dimensions or crop

        Returns:
            True if masking succeeded, False otherwise
        """
        # Named pipe functionality would be implemented here in the future
        if use_named_pipe:
            logger.debug("Named pipe functionality requested but not yet implemented")

        try:
            mask_mode = self._coerce_mask_mode(mode)
            region = self._resolve_mask_region(input_video, mask_config)
            if region is None:
                return False

            vf = self._build_video_filter(region, mask_mode)
            if not vf:
                return self._copy_video_stream(input_video, output_video)

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
                "Mask ROI configured=(x=%d,y=%d,w=%d,h=%d) effective=(x=%d,y=%d,w=%d,h=%d) input_dimensions=%dx%d mode=%s",
                region.configured_x,
                region.configured_y,
                region.configured_width,
                region.configured_height,
                region.x,
                region.y,
                region.width,
                region.height,
                region.image_width,
                region.image_height,
                mask_mode.value,
            )
            logger.info("Using mask filter: %s", vf)
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
                return self._verify_output_dimensions(output_video, region, mask_mode)
            else:
                logger.error("Masked video is empty or missing")
                return False

        except subprocess.CalledProcessError as e:
            logger.error("Streaming mask failed: %s", e.stderr)
            return False
        except (OSError, IOError) as e:
            logger.error("File operation failed during masking: %s", e)
            return False

    @staticmethod
    def _coerce_int(value: object, fallback: int) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if not isinstance(value, str):
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _canonicalize_mask_config(
        mask_config: Mapping[str, object] | None,
    ) -> dict[str, object]:
        canonical = dict(mask_config or {})
        for canonical_key, legacy_key in (
            ("x", "endoscope_image_x"),
            ("y", "endoscope_image_y"),
            ("width", "endoscope_image_width"),
            ("height", "endoscope_image_height"),
        ):
            value = canonical.get(canonical_key)
            if value is None:
                value = canonical.get(legacy_key)
            if value is not None:
                canonical[canonical_key] = value
            canonical.pop(legacy_key, None)
        return canonical

    @staticmethod
    def _coerce_mask_mode(mode: MaskMode | str) -> MaskMode:
        if isinstance(mode, MaskMode):
            return mode
        return MaskMode(str(mode))

    def _resolve_mask_region(
        self,
        input_video: Path,
        mask_config: Mapping[str, object] | None,
    ) -> VideoMaskRegionCore | None:
        normalized_config = self._canonicalize_mask_config(mask_config)
        effective_config = self.default_mask_config.model_dump()
        effective_config.update(normalized_config)

        config_image_width = self._coerce_int(
            effective_config.get("image_width"), self.default_mask_config.image_width
        )
        config_image_height = self._coerce_int(
            effective_config.get("image_height"), self.default_mask_config.image_height
        )
        configured_x = self._coerce_int(
            effective_config.get("x"), self.default_mask_config.x
        )
        configured_y = self._coerce_int(
            effective_config.get("y"), self.default_mask_config.y
        )
        configured_w = self._coerce_int(
            effective_config.get("width"), self.default_mask_config.width
        )
        configured_h = self._coerce_int(
            effective_config.get("height"), self.default_mask_config.height
        )

        format_info = video_utils.detect_video_format(input_video)
        image_width = self._coerce_int(format_info.get("width"), 0)
        image_height = self._coerce_int(format_info.get("height"), 0)
        if image_width <= 0 or image_height <= 0:
            image_width = config_image_width
            image_height = config_image_height

        if image_width <= 0 or image_height <= 0:
            logger.error("Input video dimensions could not be determined")
            return None

        if (
            config_image_width > 0
            and config_image_height > 0
            and (
                config_image_width != image_width or config_image_height != image_height
            )
        ):
            x_ratio = image_width / config_image_width
            y_ratio = image_height / config_image_height
            scaled_x = int(round(configured_x * x_ratio))
            scaled_y = int(round(configured_y * y_ratio))
            scaled_w = int(round(configured_w * x_ratio))
            scaled_h = int(round(configured_h * y_ratio))
            logger.info(
                "Scaling mask ROI from configured %dx%d to actual %dx%d: x=%d->%d y=%d->%d w=%d->%d h=%d->%d",
                config_image_width,
                config_image_height,
                image_width,
                image_height,
                configured_x,
                scaled_x,
                configured_y,
                scaled_y,
                configured_w,
                scaled_w,
                configured_h,
                scaled_h,
            )
            endoscope_x, endoscope_y = scaled_x, scaled_y
            endoscope_w, endoscope_h = scaled_w, scaled_h
        else:
            endoscope_x = configured_x
            endoscope_y = configured_y
            endoscope_w = configured_w
            endoscope_h = configured_h

        if endoscope_w <= 0 or endoscope_h <= 0:
            logger.error(
                "Invalid mask size: width=%d height=%d", endoscope_w, endoscope_h
            )
            return None

        mask_x = max(0, endoscope_x)
        mask_y = max(0, endoscope_y)
        mask_w = endoscope_w
        mask_h = endoscope_h

        if mask_x >= image_width:
            logger.error("Mask x=%d exceeds image width=%d", mask_x, image_width)
            return None
        max_mask_w = image_width - mask_x
        if mask_w > max_mask_w:
            logger.warning(
                "Mask width %d exceeds image width; clamping to %d",
                mask_w,
                max_mask_w,
            )
            mask_w = max_mask_w

        if mask_y >= image_height:
            logger.error("Mask y=%d exceeds image height=%d", mask_y, image_height)
            return None
        max_mask_h = image_height - mask_y
        if mask_h > max_mask_h:
            logger.warning(
                "Mask height %d exceeds image height; clamping to %d",
                mask_h,
                max_mask_h,
            )
            mask_h = max_mask_h

        if mask_w <= 0 or mask_h <= 0:
            logger.error(
                "Mask area collapsed: x=%d y=%d w=%d h=%d",
                mask_x,
                mask_y,
                mask_w,
                mask_h,
            )
            return None

        return VideoMaskRegionCore(
            x=mask_x,
            y=mask_y,
            width=mask_w,
            height=mask_h,
            image_width=image_width,
            image_height=image_height,
            configured_x=configured_x,
            configured_y=configured_y,
            configured_width=configured_w,
            configured_height=configured_h,
        )

    @staticmethod
    def _align_region_for_crop(
        region: VideoMaskRegionCore,
    ) -> VideoMaskRegionCore | None:
        crop_x = region.x
        crop_y = region.y
        crop_w = region.width
        crop_h = region.height

        if crop_x % 2 != 0:
            crop_x += 1
            crop_w -= 1
            logger.debug("Adjusted crop x to even boundary: x=%d w=%d", crop_x, crop_w)
        if crop_y % 2 != 0:
            crop_y += 1
            crop_h -= 1
            logger.debug("Adjusted crop y to even boundary: y=%d h=%d", crop_y, crop_h)
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
            return None

        return region.model_copy(
            update={"x": crop_x, "y": crop_y, "width": crop_w, "height": crop_h}
        )

    def _build_video_filter(self, region: VideoMaskRegionCore, mode: MaskMode) -> str:
        if mode == MaskMode.CROP:
            crop_region = self._align_region_for_crop(region)
            if crop_region is None:
                return ""
            if (
                crop_region.x == 0
                and crop_region.y == 0
                and crop_region.width == region.image_width
                and crop_region.height == region.image_height
            ):
                logger.warning("No cropping needed - endoscope covers entire frame")
                return ""
            return f"crop={crop_region.width}:{crop_region.height}:{crop_region.x}:{crop_region.y}"

        filters: list[str] = []
        if region.x > 0:
            filters.append(f"drawbox=0:0:{region.x}:ih:color=black@1:t=fill")
        right_x = region.x + region.width
        if right_x < region.image_width:
            filters.append(f"drawbox={right_x}:0:iw-{right_x}:ih:color=black@1:t=fill")
        if region.y > 0:
            filters.append(f"drawbox=0:0:iw:{region.y}:color=black@1:t=fill")
        bottom_y = region.y + region.height
        if bottom_y < region.image_height:
            filters.append(
                f"drawbox=0:{bottom_y}:iw:ih-{bottom_y}:color=black@1:t=fill"
            )

        if not filters:
            logger.warning("No masking needed - endoscope ROI covers entire frame")
            return ""
        return ",".join(filters)

    @staticmethod
    def _copy_video_stream(input_video: Path, output_video: Path) -> bool:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(input_video),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_video.exists() and output_video.stat().st_size > 0
        except subprocess.CalledProcessError as exc:
            logger.error("Video stream copy failed: %s", exc.stderr)
            return False

    def _verify_output_dimensions(
        self,
        output_video: Path,
        region: VideoMaskRegionCore,
        mode: MaskMode,
    ) -> bool:
        output_info = video_utils.detect_video_format(output_video)
        out_w = self._coerce_int(output_info.get("width"), 0)
        out_h = self._coerce_int(output_info.get("height"), 0)
        if out_w <= 0 or out_h <= 0:
            logger.warning("Could not verify masked video dimensions")
            return True

        if mode == MaskMode.PRESERVE_DIMENSIONS:
            expected_w, expected_h = region.image_width, region.image_height
        else:
            crop_region = self._align_region_for_crop(region)
            if crop_region is None:
                return False
            expected_w, expected_h = crop_region.width, crop_region.height

        if (out_w, out_h) != (expected_w, expected_h):
            logger.warning(
                "Masked video dimensions %dx%d do not match expected %dx%d",
                out_w,
                out_h,
                expected_w,
                expected_h,
            )
        return True

    def backfill_preserved_dimensions(
        self,
        *,
        source_video: Path,
        anonymized_video: Path,
        mask_config: Mapping[str, object],
        dry_run: bool = False,
    ) -> DimensionBackfillResult:
        """
        Repair a cropped anonymized video by regenerating it from the source with
        dimension-preserving black-box masking.
        """
        source_info = video_utils.detect_video_format(source_video)
        anonymized_info = video_utils.detect_video_format(anonymized_video)
        source_dimensions = (
            self._coerce_int(source_info.get("width"), 0),
            self._coerce_int(source_info.get("height"), 0),
        )
        anonymized_dimensions = (
            self._coerce_int(anonymized_info.get("width"), 0),
            self._coerce_int(anonymized_info.get("height"), 0),
        )

        if source_dimensions[0] <= 0 or source_dimensions[1] <= 0:
            return DimensionBackfillResult(
                status="source_unprobeable",
                source_dimensions=source_dimensions,
                anonymized_dimensions=anonymized_dimensions,
                detail=str(source_video),
            )
        if anonymized_dimensions[0] <= 0 or anonymized_dimensions[1] <= 0:
            return DimensionBackfillResult(
                status="anonymized_unprobeable",
                source_dimensions=source_dimensions,
                anonymized_dimensions=anonymized_dimensions,
                detail=str(anonymized_video),
            )
        if source_dimensions == anonymized_dimensions:
            return DimensionBackfillResult(
                status="already_valid",
                source_dimensions=source_dimensions,
                anonymized_dimensions=anonymized_dimensions,
            )
        if dry_run:
            return DimensionBackfillResult(
                status="would_repair",
                source_dimensions=source_dimensions,
                anonymized_dimensions=anonymized_dimensions,
            )

        temp_output = anonymized_video.with_name(
            f"{anonymized_video.stem}.dimension-backfill.{os.getpid()}{anonymized_video.suffix}"
        )
        try:
            ok = self.mask_video_streaming(
                input_video=source_video,
                mask_config=mask_config,
                output_video=temp_output,
                mode=MaskMode.PRESERVE_DIMENSIONS,
            )
            if not ok:
                return DimensionBackfillResult(
                    status="repair_failed",
                    source_dimensions=source_dimensions,
                    anonymized_dimensions=anonymized_dimensions,
                    detail="mask_video_streaming returned false",
                )
            repaired_info = video_utils.detect_video_format(temp_output)
            repaired_dimensions = (
                self._coerce_int(repaired_info.get("width"), 0),
                self._coerce_int(repaired_info.get("height"), 0),
            )
            if repaired_dimensions != source_dimensions:
                temp_output.unlink(missing_ok=True)
                return DimensionBackfillResult(
                    status="repair_dimension_mismatch",
                    source_dimensions=source_dimensions,
                    anonymized_dimensions=anonymized_dimensions,
                    detail=f"repaired_dimensions={repaired_dimensions}",
                )
            os.replace(temp_output, anonymized_video)
            return DimensionBackfillResult(
                status="repaired",
                source_dimensions=source_dimensions,
                anonymized_dimensions=anonymized_dimensions,
                repaired=True,
            )
        finally:
            temp_output.unlink(missing_ok=True)

    def create_mask_config_from_roi(
        self, endoscope_image_roi: Mapping[str, object]
    ) -> VideoMaskConfig:
        """
        Create mask config dictionary from ROI.
        Args:
            endoscope_image_roi: ROI dictionary
        Returns:
            Mask config dictionary
        """
        return VideoMaskConfig.model_validate(
            {
                "x": endoscope_image_roi["x"],
                "y": endoscope_image_roi["y"],
                "width": endoscope_image_roi["width"],
                "height": endoscope_image_roi["height"],
                "image_width": endoscope_image_roi.get(
                    "image_width", self.default_mask_config.image_width
                ),
                "image_height": endoscope_image_roi.get(
                    "image_height", self.default_mask_config.image_height
                ),
            }
        )

    def _load_mask(self) -> VideoMaskConfig:
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
                loaded = cast(dict[str, object], json.load(f))
                normalized = self._canonicalize_mask_config(loaded)
                normalized.pop("description", None)
                return VideoMaskConfig.model_validate(normalized)
        except (FileNotFoundError, json.JSONDecodeError):
            # create or overwrite with a fresh stub
            masks_dir.mkdir(parents=True, exist_ok=True)
            with mask_file.open("w") as f:
                json.dump(stub, f, indent=2)
            logger.warning(
                "Created or repaired mask file %s – please verify coordinates.",
                mask_file,
            )
            return VideoMaskConfig.model_validate(stub)

        except (json.JSONDecodeError, IOError) as e:
            logger.error(
                f"Failed to load/create mask configuration for {self.device_name}: {e}"
            )
            raise FileNotFoundError(
                f"Could not load or create mask configuration for {self.device_name}: {e}"
            )
