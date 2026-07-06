from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import fitz  # type: ignore[import-untyped, reportMissingTypeStubs]  # PyMuPDF
import pytest
from PIL import Image

from lx_anonymizer.anonymization.anonymizer import Anonymizer

Roi = tuple[int, int, int, int]
TextWithBox = dict[str, str | Roi]


def create_test_image(
    path: Path, size: tuple[int, int] = (200, 100), color: str = "white"
) -> None:
    image = Image.new("RGB", size, color=color)
    image.save(path)


def create_test_pdf_with_text(
    path: Path, text: str = "Patient: Max Mustermann"
) -> None:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 100), text, fontsize=12)  # pyright: ignore[reportUnknownMemberType]
    doc.save(path)  # pyright: ignore[reportUnknownMemberType]
    doc.close()


def create_test_pdf_with_multiple_pages(path: Path) -> None:
    doc = fitz.open()

    page1 = doc.new_page(width=300, height=200)
    page1.insert_text((50, 100), "Patient: Alice Example", fontsize=12)  # pyright: ignore[reportUnknownMemberType]

    page2 = doc.new_page(width=300, height=200)
    page2.insert_text((50, 100), "Patient: Bob Example", fontsize=12)  # pyright: ignore[reportUnknownMemberType]

    doc.save(path)  # pyright: ignore[reportUnknownMemberType]
    doc.close()


def _blank_pdf_page(_pdf_path: object) -> list[Image.Image]:
    return [Image.new("RGB", (300, 200), "white")]


def _blank_pdf_pages(_pdf_path: object) -> list[Image.Image]:
    return [
        Image.new("RGB", (300, 200), "white"),
        Image.new("RGB", (300, 200), "white"),
    ]


def _no_sensitive_regions(
    _self: Anonymizer, _image: Image.Image, **_kwargs: object
) -> list[Roi]:
    return []


def _sample_sensitive_region(
    _self: Anonymizer, _image: Image.Image, **_kwargs: object
) -> list[Roi]:
    return [(10, 10, 40, 30)]


def _pdf_text_region(
    _self: Anonymizer, _image: Image.Image, **_kwargs: object
) -> list[Roi]:
    return [(40, 80, 220, 110)]


def _east_one_box(
    _image: Image.Image, _min_confidence: float, _width: int, _height: int
) -> tuple[list[Roi], str]:
    return ([(1, 2, 3, 4)], "[]")


def _east_no_boxes(
    _image: Image.Image, _min_confidence: float, _width: int, _height: int
) -> tuple[list[Roi], str]:
    return ([], "[]")


def _ocr_one_box(
    _image: Image.Image, _text_boxes: Sequence[Roi], language: str = "deu+eng"
) -> tuple[list[TextWithBox], list[Roi]]:
    _ = language
    return ([{"text": "Max", "box": (1, 2, 3, 4)}], [])


def _phi_region(_image: Image.Image) -> list[Roi]:
    return [(40, 50, 120, 90)]


def _cropper_region(
    _image: Image.Image, _text_with_boxes: Sequence[TextWithBox]
) -> list[Roi]:
    return [(10, 10, 20, 20)]


def _raise_pdf_open(_path: object) -> object:
    raise RuntimeError("cannot open pdf")


@pytest.fixture
def anonymizer() -> Anonymizer:
    return Anonymizer()


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    create_test_image(path)
    return path


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    create_test_pdf_with_text(path)
    return path


@pytest.fixture
def sample_pdf_multi_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample_multi.pdf"
    create_test_pdf_with_multiple_pages(path)
    return path


def test_create_anonymized_image_saves_output_even_when_no_text_detected(
    anonymizer: Anonymizer,
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "anonymized.png"

    monkeypatch.setattr(
        Anonymizer,
        "_detect_sensitive_regions_from_image",
        _no_sensitive_regions,
    )

    result = anonymizer.create_anonymized_image(
        str(sample_image_path),
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    original = Image.open(sample_image_path).convert("RGB")
    anonymized = Image.open(output_path).convert("RGB")
    assert original.tobytes() == anonymized.tobytes()


def test_create_anonymized_image_blackens_sensitive_regions(
    anonymizer: Anonymizer,
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "anonymized.png"

    monkeypatch.setattr(
        Anonymizer,
        "_detect_sensitive_regions_from_image",
        _sample_sensitive_region,
    )

    result = anonymizer.create_anonymized_image(
        str(sample_image_path),
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    anonymized = Image.open(output_path).convert("RGB")

    # A pixel well inside the ROI should be black.
    assert anonymized.getpixel((20, 20)) == (0, 0, 0)

    # A pixel clearly outside the ROI should remain white.
    assert anonymized.getpixel((100, 50)) == (255, 255, 255)


def test_create_anonymized_image_from_rois_blackens_regions(
    anonymizer: Anonymizer,
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "anonymized_roi.png"

    result = anonymizer.create_anonymized_image_from_rois(
        str(sample_image_path),
        rois=[(20, 20, 60, 40)],
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    anonymized = Image.open(output_path).convert("RGB")
    assert anonymized.getpixel((30, 30)) == (0, 0, 0)
    assert anonymized.getpixel((150, 50)) == (255, 255, 255)


def test_create_anonymized_image_from_rois_with_empty_rois_still_saves(
    anonymizer: Anonymizer,
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "anonymized_empty.png"

    result = anonymizer.create_anonymized_image_from_rois(
        str(sample_image_path),
        rois=[],
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    original = Image.open(sample_image_path).convert("RGB")
    anonymized = Image.open(output_path).convert("RGB")
    assert original.tobytes() == anonymized.tobytes()


def test_create_anonymized_image_from_rois_clamps_out_of_bounds_boxes(
    anonymizer: Anonymizer,
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "anonymized_clamped.png"

    result = anonymizer.create_anonymized_image_from_rois(
        str(sample_image_path),
        rois=[(-50, -50, 50, 50)],
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    anonymized = Image.open(output_path).convert("RGB")

    # Near top-left should be black because ROI gets clamped into image bounds.
    assert anonymized.getpixel((10, 10)) == (0, 0, 0)

    # Far away should still be white.
    assert anonymized.getpixel((150, 80)) == (255, 255, 255)


def test_create_anonymized_pdf_saves_output_even_when_no_sensitive_regions_found(
    anonymizer: Anonymizer,
    sample_pdf_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "anonymized.pdf"

    # Return one synthetic raster page so the pipeline has something to iterate over.
    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images",
        _blank_pdf_page,
    )

    monkeypatch.setattr(
        Anonymizer,
        "_detect_sensitive_regions_from_image",
        _no_sensitive_regions,
    )

    result = anonymizer.create_anonymized_pdf(
        str(sample_pdf_path),
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    doc = fitz.open(output_path)
    try:
        text = cast(str, doc[0].get_text())  # pyright: ignore[reportUnknownMemberType]
    finally:
        doc.close()

    assert "Patient: Max Mustermann" in text


def test_create_anonymized_pdf_from_rois_removes_text_in_redacted_region(
    anonymizer: Anonymizer,
    sample_pdf_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "redacted.pdf"

    # Match the PDF page size exactly to keep coordinate mapping simple.
    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images",
        _blank_pdf_page,
    )

    # The text is inserted around (50, 100), so redact that region.
    rois_per_page = {
        0: [(40, 80, 220, 110)],
    }

    result = anonymizer.create_anonymized_pdf_from_rois(
        str(sample_pdf_path),
        rois_per_page=rois_per_page,
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    doc = fitz.open(output_path)
    try:
        text = cast(str, doc[0].get_text())  # pyright: ignore[reportUnknownMemberType]
    finally:
        doc.close()

    # This is the crucial anonymization test:
    # text should no longer be extractable from the redacted output.
    assert "Max Mustermann" not in text
    assert "Patient:" not in text


def test_create_anonymized_pdf_from_rois_only_redacts_target_page(
    anonymizer: Anonymizer,
    sample_pdf_multi_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "redacted_multi.pdf"

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images",
        _blank_pdf_pages,
    )

    rois_per_page = {
        0: [(40, 80, 220, 110)],
    }

    result = anonymizer.create_anonymized_pdf_from_rois(
        str(sample_pdf_multi_path),
        rois_per_page=rois_per_page,
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    doc = fitz.open(output_path)
    try:
        page0_text = cast(str, doc[0].get_text())  # pyright: ignore[reportUnknownMemberType]
        page1_text = cast(str, doc[1].get_text())  # pyright: ignore[reportUnknownMemberType]
    finally:
        doc.close()

    assert "Alice Example" not in page0_text
    assert "Patient:" not in page0_text

    assert "Bob Example" in page1_text
    assert "Patient:" in page1_text


def test_create_anonymized_pdf_uses_detected_sensitive_regions(
    anonymizer: Anonymizer,
    sample_pdf_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "detected_redacted.pdf"

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images",
        _blank_pdf_page,
    )

    monkeypatch.setattr(
        Anonymizer,
        "_detect_sensitive_regions_from_image",
        _pdf_text_region,
    )

    result = anonymizer.create_anonymized_pdf(
        str(sample_pdf_path),
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    doc = fitz.open(output_path)
    try:
        text = cast(str, doc[0].get_text())  # pyright: ignore[reportUnknownMemberType]
    finally:
        doc.close()

    assert "Max Mustermann" not in text
    assert "Patient:" not in text


def test_create_anonymized_pdf_from_rois_with_empty_mapping_preserves_text(
    anonymizer: Anonymizer,
    sample_pdf_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "unchanged.pdf"

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images",
        _blank_pdf_page,
    )

    result = anonymizer.create_anonymized_pdf_from_rois(
        str(sample_pdf_path),
        rois_per_page={},
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    doc = fitz.open(output_path)
    try:
        text = cast(str, doc[0].get_text())  # pyright: ignore[reportUnknownMemberType]
    finally:
        doc.close()

    assert "Patient: Max Mustermann" in text


def test_detect_sensitive_regions_pipeline_calls_cropper_with_ocr_output(
    anonymizer: Anonymizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (300, 200), "white")

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.east_text_detection",
        _east_one_box,
    )

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer._ocr_text_on_boxes",
        _ocr_one_box,
    )

    called: dict[str, object] = {}

    def fake_detect_sensitive_regions(
        image_arg: Image.Image, text_with_boxes: Sequence[TextWithBox]
    ) -> list[Roi]:
        called["image"] = image_arg
        called["text_with_boxes"] = text_with_boxes
        return [(10, 10, 20, 20)]

    monkeypatch.setattr(
        anonymizer.sensitive_cropper,
        "detect_sensitive_regions",
        fake_detect_sensitive_regions,
    )

    rois = anonymizer._detect_sensitive_regions_from_image(image)  # pyright: ignore[reportPrivateUsage]

    assert rois == [(10, 10, 20, 20)]
    assert called["image"] is image
    assert called["text_with_boxes"] == [{"text": "Max", "box": (1, 2, 3, 4)}]


def test_detect_sensitive_regions_includes_custom_phi_detector_regions(
    anonymizer: Anonymizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (300, 200), "white")

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.detect_phi_regions_from_settings",
        _phi_region,
    )
    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.east_text_detection",
        _east_no_boxes,
    )

    rois = anonymizer._detect_sensitive_regions_from_image(image)  # pyright: ignore[reportPrivateUsage]

    assert rois == [(40, 50, 120, 90)]


def test_detect_sensitive_regions_merges_custom_and_ocr_regions(
    anonymizer: Anonymizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (300, 200), "white")

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.detect_phi_regions_from_settings",
        _phi_region,
    )
    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.east_text_detection",
        _east_one_box,
    )
    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer._ocr_text_on_boxes",
        _ocr_one_box,
    )
    monkeypatch.setattr(
        anonymizer.sensitive_cropper,
        "detect_sensitive_regions",
        _cropper_region,
    )

    rois = anonymizer._detect_sensitive_regions_from_image(image)  # pyright: ignore[reportPrivateUsage]

    assert rois == [(10, 10, 20, 20), (40, 50, 120, 90)]


def test_create_anonymized_pdf_returns_none_on_pdf_open_failure(
    anonymizer: Anonymizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "missing.pdf"
    output_path = tmp_path / "out.pdf"

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images",
        _blank_pdf_page,
    )

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.pymupdf.open", _raise_pdf_open
    )

    result = anonymizer.create_anonymized_pdf(
        str(input_path),
        output_path=str(output_path),
    )

    assert result is None
    assert not output_path.exists()
