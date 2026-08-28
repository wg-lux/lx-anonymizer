from __future__ import annotations

import argparse
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal, overload
from uuid import uuid4

from lx_dtypes.models.contracts.image_processing import ImageProcessingResultPayload

from lx_anonymizer.hardware.gpu_management import clear_gpu_memory
from lx_anonymizer.image_processing.image_loader import get_image_paths
from lx_anonymizer.image_processing.image_processor import process_image
from lx_anonymizer.image_processing.pdf_operations import (
    convert_image_to_pdf,
    merge_pdfs,
)
from lx_anonymizer.processing_contracts import (
    ImageAnonymizationRequest,
    ImageAnonymizationResult,
)
from lx_anonymizer.setup.custom_logger import configure_global_logger, get_logger
from lx_anonymizer.setup.directory_setup import (
    create_results_directory,
    create_temp_directory,
)
from lx_anonymizer.text_detection.phi_region_detector import PhiRegionDetector

"""
Main Function

Run by calling main(image_or_pdf_path, east_path=None, device="olympus_cv_1500", validation=False, min_confidence=0.5, width=320, height=320)

Or in the directory python main.py -i /path/to/image.png -east /path/to/east_detector.pb -d olympus_cv_1500 -c 0.5 -w 320 -e 320

Disclaimer: The Pipeline is easier to run with the Agl-Anonymizer-Flake API.

Parameters:
- image_or_pdf_path: str | Path
    The path to the image or PDF file to process.
- east_path: Path
    The path to the EAST text detector model. Default is None.
- device: str
    The device name to set the correct text settings. Default is "olympus_cv_1500".
- validation: bool
    A boolean value representing if validation through the AGL-Validator is required. Default is False.
- min_confidence: float
    The minimum probability required to inspect a region. Default is 0.5.
- width: int
    The resized image width (should be a multiple of 32). Default is 320.
- height: int
    The resized image height (should be a multiple of 32). Default is 320.

Returns:
- Path
    The path to the output image or PDF file.
- tuple[Path, dict[str, object], Path]
    A tuple containing the path to the output image/PDF, anonymization metadata, and original input path.

Directory Setup:

You can change the default path for the base directory as well as the temporary directory
by changing the values specified in the directory_setup.py file.
"""


AnonymizationPayload = dict[str, object]
MainResult = Path | tuple[Path, AnonymizationPayload, Path]


@contextmanager
def temp_directory_manager() -> Generator[tuple[Path, Path, Path], None, None]:
    logger = get_logger(__name__)
    temp_dir, base_dir, csv_dir = create_temp_directory()
    try:
        yield temp_dir, base_dir, csv_dir
    finally:
        if temp_dir.exists() and temp_dir.is_dir():
            for file in temp_dir.iterdir():
                try:
                    file.unlink()
                except Exception as exc:
                    logger.warning(f"Failed to delete temp file {file}: {exc}")


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""

    pass


class ImageAnonymizer:
    """Typed image/PDF processing strand with an injectable region detector."""

    def __init__(self, region_detector: PhiRegionDetector | None = None) -> None:
        self.region_detector = region_detector

    def process(
        self,
        request: ImageAnonymizationRequest,
    ) -> ImageAnonymizationResult:
        clear_gpu_memory()
        east_model_path = request.east_model_path or Path(
            "frozen_east_text_detection.pb"
        )
        with temp_directory_manager() as (temp_dir, _base_dir, _csv_dir):
            image_paths = get_image_paths(request.source_path, temp_dir)
            processed_pdf_paths: list[Path] = []
            typed_result: ImageProcessingResultPayload | None = None

            for image_path in image_paths:
                if not image_path.is_file():
                    raise ImageProcessingError(
                        f"Image path does not exist: {image_path}"
                    )
                processed_image_path, raw_result = process_image(
                    image_path,
                    east_model_path,
                    request.device,
                    request.min_confidence,
                    request.detector_width,
                    request.detector_height,
                    request.output_directory,
                    temp_dir,
                    region_detector=self.region_detector,
                )
                typed_result = ImageProcessingResultPayload.model_validate(raw_result)
                if request.source_path.suffix.casefold() == ".pdf":
                    temporary_pdf = temp_dir / f"temporary_pdf_{uuid4()}.pdf"
                    convert_image_to_pdf(processed_image_path, temporary_pdf)
                    processed_pdf_paths.append(temporary_pdf)
                else:
                    processed_pdf_paths.append(processed_image_path)

            if not processed_pdf_paths or typed_result is None:
                raise ImageProcessingError("No processed images were generated.")

            if request.source_path.suffix.casefold() == ".pdf":
                artifact_path = (
                    request.output_directory / f"final_document_{uuid4()}.pdf"
                )
                merge_pdfs(list(processed_pdf_paths), artifact_path)
            else:
                processed_path = processed_pdf_paths[0]
                artifact_path = request.output_directory / (
                    f"final_image_{uuid4()}{request.source_path.suffix.casefold()}"
                )
                shutil.copy2(processed_path, artifact_path)

            if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
                raise ImageProcessingError(
                    f"Image pipeline did not produce a non-empty artifact: {artifact_path}"
                )
            get_logger(__name__).info("Output Path: %s", artifact_path)
            return ImageAnonymizationResult(
                source_path=request.source_path,
                artifact_path=artifact_path,
                metadata=typed_result,
            )


@overload
def main(
    image_or_pdf_path: str | Path,
    east_path: Path | None,
    device: str,
    validation: Literal[False],
    min_confidence: float,
    width: int,
    height: int,
    region_detector: PhiRegionDetector | None = None,
) -> Path: ...


@overload
def main(
    image_or_pdf_path: str | Path,
    east_path: Path | None = None,
    device: str = "olympus_cv_1500",
    validation: Literal[True] = True,
    min_confidence: float = 0.5,
    width: int = 320,
    height: int = 320,
    region_detector: PhiRegionDetector | None = None,
) -> tuple[Path, AnonymizationPayload, Path]: ...


def main(
    image_or_pdf_path: str | Path,
    east_path: Path | None = None,
    device: str = "olympus_cv_1500",
    validation: bool = False,
    min_confidence: float = 0.5,
    width: int = 320,
    height: int = 320,
    region_detector: PhiRegionDetector | None = None,
) -> MainResult:
    source_path = Path(image_or_pdf_path)

    try:
        request = ImageAnonymizationRequest(
            source_path=source_path,
            output_directory=create_results_directory(),
            east_model_path=east_path,
            device=device,
            min_confidence=min_confidence,
            detector_width=width,
            detector_height=height,
        )
        result = ImageAnonymizer(region_detector=region_detector).process(request)
        if not validation:
            return result.artifact_path
        return (
            result.artifact_path,
            result.metadata.model_dump(),
            result.source_path,
        )

    except Exception as exc:
        raise ImageProcessingError(f"Processing failed: {exc}") from exc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input image"
    )
    ap.add_argument(
        "-east",
        "--east",
        type=str,
        required=False,
        help="Path to input EAST text detector",
    )
    ap.add_argument(
        "-d",
        "--device",
        type=str,
        default="olympus_cv_1500",
        help="Device name to set the correct text settings",
    )
    ap.add_argument(
        "-V",
        "--validation",
        action="store_true",
        help="Enable validation metadata output",
    )
    ap.add_argument(
        "-c",
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum probability required to inspect a region",
    )
    ap.add_argument(
        "-w",
        "--width",
        type=int,
        default=320,
        help="Resized image width (should be multiple of 32)",
    )
    ap.add_argument(
        "-e",
        "--height",
        type=int,
        default=320,
        help="Resized image height (should be multiple of 32)",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = ap.parse_args()
    configure_global_logger(verbose=args.verbose)

    main(
        image_or_pdf_path=Path(args.image),
        east_path=Path(args.east) if args.east else None,
        device=args.device,
        validation=args.validation,
        min_confidence=args.min_confidence,
        width=args.width,
        height=args.height,
    )
