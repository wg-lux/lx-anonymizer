from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from lx_dtypes.models.contracts.image_processing import ImageProcessingResultPayload
from lx_dtypes.models.contracts.report_anonymization import ReportAnonymizationResult
from lx_dtypes.models.meta.VideoMeta import VideoMeta
from PIL import Image
from pydantic import ValidationError

from lx_anonymizer.frame_cleaner import FrameCleaner
from lx_anonymizer.main_with_reassembly import ImageAnonymizer
from lx_anonymizer.processing_contracts import (
    ImageAnonymizationRequest,
    VideoAnonymizationRequest,
)
from lx_anonymizer.report_contracts import ReportAnonymizationRequest
from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.text_detection.phi_region_detector import PhiRegionDetector


class _FixedDetector:
    def detect(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        assert image.size == (8, 8)
        return [(1, 2, 6, 7)]


@contextmanager
def _temporary_directories(
    root: Path,
) -> Generator[tuple[Path, Path, Path], None, None]:
    temporary = root / "temporary"
    base = root / "base"
    csv = base / "csv"
    temporary.mkdir()
    csv.mkdir(parents=True)
    yield temporary, base, csv


def test_image_strand_processes_a_typed_request_with_explicit_detector(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(source)
    output_directory = tmp_path / "attempt"
    output_directory.mkdir()
    detector: PhiRegionDetector = _FixedDetector()
    request = ImageAnonymizationRequest(
        source_path=source,
        output_directory=output_directory,
    )
    observed_detectors: list[PhiRegionDetector | None] = []

    def fake_process_image(
        image_path: Path,
        _east_path: Path,
        _device: str,
        _confidence: float,
        _width: int,
        _height: int,
        results_dir: Path,
        _temp_dir: Path,
        text_extracted: bool = False,
        skip_blur: bool = False,
        skip_reassembly: bool = False,
        disable_llm: bool = False,
        region_detector: PhiRegionDetector | None = None,
    ) -> tuple[Path, dict[str, object]]:
        del text_extracted, skip_blur, skip_reassembly, disable_llm
        observed_detectors.append(region_detector)
        processed = results_dir / "processed.png"
        processed.write_bytes(image_path.read_bytes())
        payload = ImageProcessingResultPayload(
            filename=image_path,
            file_type="png",
            extracted_text="",
        )
        return processed, payload.model_dump()

    with (
        patch("lx_anonymizer.main_with_reassembly.clear_gpu_memory"),
        patch(
            "lx_anonymizer.main_with_reassembly.temp_directory_manager",
            return_value=_temporary_directories(tmp_path),
        ),
        patch(
            "lx_anonymizer.main_with_reassembly.get_image_paths",
            return_value=[source],
        ),
        patch(
            "lx_anonymizer.main_with_reassembly.process_image",
            side_effect=fake_process_image,
        ),
    ):
        result = ImageAnonymizer(region_detector=detector).process(request)

    assert observed_detectors == [detector]
    assert result.source_path == source.resolve()
    assert result.artifact_path.parent == output_directory.resolve()
    assert result.artifact_path.read_bytes() == source.read_bytes()
    assert result.metadata.file_type == "png"


@pytest.mark.parametrize("dimension", [1, 31, 33, 100])
def test_image_request_rejects_non_aligned_detector_dimensions(
    tmp_path: Path,
    dimension: int,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    output_directory = tmp_path / "attempt"
    output_directory.mkdir()
    with pytest.raises(ValidationError, match="multiples of 32"):
        ImageAnonymizationRequest(
            source_path=source,
            output_directory=output_directory,
            detector_width=dimension,
        )


def test_frame_cleaner_process_wraps_the_legacy_result_in_typed_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "candidate.mp4"
    request = VideoAnonymizationRequest(
        source_path=source,
        output_path=output,
        source_frame_rate=Fraction(25, 1),
    )
    cleaner = object.__new__(FrameCleaner)

    def fake_clean_video(**_kwargs: object) -> tuple[Path, dict[str, object]]:
        output.write_bytes(b"candidate")
        return output, VideoMeta(file_path=str(source)).model_dump(mode="json")

    with patch.object(cleaner, "clean_video", side_effect=fake_clean_video):
        result = cleaner.process(request)

    assert result.source_path == source.resolve()
    assert result.artifact_path == output
    assert isinstance(result.metadata, VideoMeta)


def test_report_reader_process_is_the_consistent_alias() -> None:
    reader = object.__new__(ReportReader)
    request = cast(ReportAnonymizationRequest, object())
    expected = cast(ReportAnonymizationResult, object())
    with patch.object(
        reader, "process_report", return_value=expected
    ) as process_report:
        assert reader.process(request) is expected
    process_report.assert_called_once_with(request)
