from __future__ import annotations

import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
from PIL import Image

from lx_anonymizer.setup.custom_logger import logger
from lx_anonymizer.text_detection.phi_region_detector import (
    detect_phi_regions_from_settings,
)

Box = tuple[int, int, int, int]
RegionDetector = Callable[[Image.Image], list[Box]]


@dataclass(frozen=True)
class DetectorVideoMaskingSummary:
    frames_processed: int
    frames_with_redactions: int
    redactions_applied: int
    output_path: str
    strategy: str = "detector_assisted"
    review_required: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DetectorVideoMasker:
    """Apply configured PHI-detector regions to every decoded video frame."""

    def __init__(self, detector: RegionDetector | None = None) -> None:
        self._detector = detector or detect_phi_regions_from_settings

    def mask_video(
        self, input_video: Path, output_video: Path
    ) -> DetectorVideoMaskingSummary:
        input_video = input_video.expanduser().resolve()
        output_video = output_video.expanduser().resolve()
        if not input_video.is_file():
            raise FileNotFoundError(f"Input video not found: {input_video}")

        output_video.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="lx-anonymizer-detector-mask-", dir=output_video.parent
        ) as temp_dir:
            video_only = Path(temp_dir) / "masked-video.mp4"
            summary = self._mask_frames(input_video, video_only)
            self._mux_original_audio(video_only, input_video, output_video)

        if not output_video.is_file() or output_video.stat().st_size <= 0:
            raise RuntimeError("Detector-assisted video output is missing or empty")
        return DetectorVideoMaskingSummary(
            frames_processed=summary.frames_processed,
            frames_with_redactions=summary.frames_with_redactions,
            redactions_applied=summary.redactions_applied,
            output_path=str(output_video),
        )

    def _mask_frames(
        self, input_video: Path, video_only_output: Path
    ) -> DetectorVideoMaskingSummary:
        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not decode input video: {input_video}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # pyright: ignore[reportUnknownMemberType]
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # pyright: ignore[reportUnknownMemberType]
        fps = float(capture.get(cv2.CAP_PROP_FPS))  # pyright: ignore[reportUnknownMemberType]
        if width <= 0 or height <= 0 or fps <= 0:
            capture.release()
            raise RuntimeError("Input video has invalid dimensions or frame rate")

        writer = cv2.VideoWriter(
            str(video_only_output),
            cv2.VideoWriter_fourcc(*"mp4v"),  # pyright: ignore[reportUnknownMemberType]
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not initialize detector-assisted video writer")

        frames_processed = 0
        frames_with_redactions = 0
        redactions_applied = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                regions = self._detector(image)
                valid_regions = 0
                for x1, y1, x2, y2 in regions:
                    left = max(0, min(int(x1), width))
                    top = max(0, min(int(y1), height))
                    right = max(0, min(int(x2), width))
                    bottom = max(0, min(int(y2), height))
                    if right <= left or bottom <= top:
                        continue
                    frame[top:bottom, left:right] = 0
                    valid_regions += 1
                if valid_regions:
                    frames_with_redactions += 1
                    redactions_applied += valid_regions
                writer.write(frame)  # pyright: ignore[reportUnknownMemberType]
                frames_processed += 1
        finally:
            capture.release()
            writer.release()

        if frames_processed == 0:
            raise RuntimeError("Input video contained no decodable frames")
        if not video_only_output.is_file() or video_only_output.stat().st_size <= 0:
            raise RuntimeError("Frame masking produced no video output")

        logger.info(
            "Detector-assisted masking processed %d frames and applied %d regions",
            frames_processed,
            redactions_applied,
        )
        return DetectorVideoMaskingSummary(
            frames_processed=frames_processed,
            frames_with_redactions=frames_with_redactions,
            redactions_applied=redactions_applied,
            output_path=str(video_only_output),
        )

    @staticmethod
    def _mux_original_audio(
        masked_video: Path, original_video: Path, output_video: Path
    ) -> None:
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(masked_video),
            "-i",
            str(original_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Could not finalize detector-assisted video: {exc.stderr}"
            ) from exc
