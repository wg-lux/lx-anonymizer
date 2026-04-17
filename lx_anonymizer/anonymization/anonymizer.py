from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pymupdf  # type: ignore[import-untyped]
from PIL import Image, ImageDraw

from lx_anonymizer.anonymization.sensitive_region_cropper import SensitiveRegionCropper
from lx_anonymizer.image_processing.pdf_operations import convert_pdf_to_images
from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.text_detection.east_text_detection import east_text_detection

logger = get_logger(__name__)


def _ocr_text_on_boxes(image: Image.Image, text_boxes, language: str = "deu+eng"):
    try:
        from lx_anonymizer.ocr.ocr_tesserocr import tesseract_on_boxes_fast

        return tesseract_on_boxes_fast(image, text_boxes, language=language)
    except ImportError:
        from lx_anonymizer.ocr.ocr import tesseract_on_boxes_pytesseract

        logger.info("tesserocr not available; falling back to pytesseract OCR")
        return tesseract_on_boxes_pytesseract(image, text_boxes)


def _east_text_detection_on_pil_image(
    image: Image.Image,
    min_confidence: float,
    width: int,
    height: int,
) -> tuple[list[tuple[int, int, int, int]], str]:
    """
    Temporary adapter for east_text_detection(), which currently expects str | Path.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        image.save(tmp_path)
        return east_text_detection(
            tmp_path,
            min_confidence=min_confidence,
            width=width,
            height=height,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove temporary EAST image: %s", tmp_path)


def _clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    padding: int = 0,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(x1 - padding, width))
    y1 = max(0, min(y1 - padding, height))
    x2 = max(0, min(x2 + padding, width))
    y2 = max(0, min(y2 + padding, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _pil_to_pdf_rect(
    box: tuple[int, int, int, int],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
    padding: float = 2.0,
) -> pymupdf.Rect | None:
    x1, y1, x2, y2 = box

    if image_width <= 0 or image_height <= 0:
        return None

    scale_x = page_width / image_width
    scale_y = page_height / image_height

    pdf_x1 = x1 * scale_x
    pdf_y1 = page_height - (y2 * scale_y)
    pdf_x2 = x2 * scale_x
    pdf_y2 = page_height - (y1 * scale_y)

    rect = pymupdf.Rect(
        pdf_x1 - padding,
        pdf_y1 - padding,
        pdf_x2 + padding,
        pdf_y2 + padding,
    )

    page_bounds = pymupdf.Rect(0, 0, page_width, page_height)
    rect = rect & page_bounds

    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None

    return rect


class Anonymizer:
    def __init__(self) -> None:
        self.sensitive_cropper = SensitiveRegionCropper()

    def _default_output_path(self, input_path: str, suffix: str = "_anonymized") -> str:
        path = Path(input_path)
        return str(path.with_stem(path.stem + suffix))

    def _ensure_output_parent(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _detect_sensitive_regions_from_image(
        self,
        image: Image.Image,
        east_min_confidence: float = 0.5,
        east_width: int = 640,
        east_height: int = 640,
        language: str = "deu+eng",
    ) -> list[tuple[int, int, int, int]]:
        logger.debug("Running EAST text detection")
        text_boxes, _ = _east_text_detection_on_pil_image(
            image,
            min_confidence=east_min_confidence,
            width=east_width,
            height=east_height,
        )

        if not text_boxes:
            logger.info("No text regions detected")
            return []

        logger.info("EAST detected %d text regions", len(text_boxes))

        logger.debug("Running OCR on %d boxes", len(text_boxes))
        text_with_boxes, _ = _ocr_text_on_boxes(
            image,
            text_boxes,
            language=language,
        )

        sensitive_regions = self.sensitive_cropper.detect_sensitive_regions(
            image,
            text_with_boxes,
        )

        if sensitive_regions:
            logger.info("Detected %d sensitive regions", len(sensitive_regions))
        else:
            logger.info("No sensitive regions detected after OCR/classification")

        return sensitive_regions

    def _draw_black_boxes_on_image(
        self,
        image: Image.Image,
        rois: list[tuple[int, int, int, int]],
        padding: int = 2,
    ) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for roi in rois:
            clamped = _clamp_box(*roi, width=width, height=height, padding=padding)
            if clamped is None:
                logger.debug("Skipping invalid ROI for image draw: %s", roi)
                continue

            x1, y1, x2, y2 = clamped
            draw.rectangle([(x1, y1), (x2, y2)], fill="black")
            logger.debug("Blackened image ROI: (%d, %d, %d, %d)", x1, y1, x2, y2)

        return image

    def _apply_pdf_redactions(
        self,
        doc: pymupdf.Document,
        images: list[Image.Image],
        rois_per_page: dict[int, list[tuple[int, int, int, int]]],
        padding_points: float = 2.0,
    ) -> None:
        for page_num, image in enumerate(images):
            page = doc[page_num]
            rois = rois_per_page.get(page_num, [])
            if not rois:
                continue

            logger.info("Redacting %d regions on page %d", len(rois), page_num + 1)

            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            image_width, image_height = image.size

            added = 0
            for roi in rois:
                clamped = _clamp_box(
                    *roi,
                    width=image_width,
                    height=image_height,
                    padding=0,
                )
                if clamped is None:
                    logger.debug("Skipping invalid PDF ROI: %s", roi)
                    continue

                rect = _pil_to_pdf_rect(
                    clamped,
                    page_width=page_width,
                    page_height=page_height,
                    image_width=image_width,
                    image_height=image_height,
                    padding=padding_points,
                )
                if rect is None:
                    logger.debug("Skipping empty converted PDF rect for ROI: %s", roi)
                    continue

                page.add_redact_annot(rect, fill=(0, 0, 0))
                added += 1

            if added:
                # This permanently removes content within the redaction areas.
                page.apply_redactions()

    def create_anonymized_pdf(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        report_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        _ = report_meta

        try:
            if output_path is None:
                output_path = self._default_output_path(pdf_path)

            self._ensure_output_parent(output_path)
            logger.info(
                "Creating anonymized PDF with EAST + OCR redaction: %s", output_path
            )

            images = convert_pdf_to_images(pdf_path)
            rois_per_page: dict[int, list[tuple[int, int, int, int]]] = {}

            for page_num, image in enumerate(images):
                logger.debug("Processing PDF page %d", page_num + 1)
                rois = self._detect_sensitive_regions_from_image(image)
                rois_per_page[page_num] = rois

            doc = pymupdf.open(pdf_path)
            try:
                self._apply_pdf_redactions(doc, images, rois_per_page)
                doc.save(output_path)
            finally:
                doc.close()

            logger.info("Anonymized PDF saved: %s", output_path)
            return output_path

        except ImportError as exc:
            logger.error(
                "Required module not installed. Cannot create anonymized PDF: %s", exc
            )
            return None
        except Exception as exc:
            logger.error("Error creating anonymized PDF: %s", exc, exc_info=True)
            return None

    def create_anonymized_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        report_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        _ = report_meta

        try:
            if output_path is None:
                output_path = self._default_output_path(image_path)

            self._ensure_output_parent(output_path)
            logger.info("Creating anonymized image with EAST + OCR: %s", output_path)

            image = Image.open(image_path).convert("RGB")
            rois = self._detect_sensitive_regions_from_image(image)

            anonymized = self._draw_black_boxes_on_image(image, rois, padding=2)
            anonymized.save(output_path)

            logger.info("Anonymized image saved: %s", output_path)
            return output_path

        except ImportError as exc:
            logger.error(
                "Required module not installed. Cannot create anonymized image: %s", exc
            )
            return None
        except Exception as exc:
            logger.error("Error creating anonymized image: %s", exc, exc_info=True)
            return None

    def create_anonymized_pdf_from_rois(
        self,
        pdf_path: str,
        rois_per_page: dict[int, list[tuple[int, int, int, int]]],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        try:
            if output_path is None:
                output_path = self._default_output_path(pdf_path)

            self._ensure_output_parent(output_path)
            logger.info("Creating anonymized PDF from ROIs: %s", output_path)

            images = convert_pdf_to_images(pdf_path)

            doc = pymupdf.open(pdf_path)
            try:
                self._apply_pdf_redactions(doc, images, rois_per_page)
                doc.save(output_path)
            finally:
                doc.close()

            logger.info("Anonymized PDF saved: %s", output_path)
            return output_path

        except Exception as exc:
            logger.error(
                "Error creating anonymized PDF from ROIs: %s", exc, exc_info=True
            )
            return None

    def create_anonymized_image_from_rois(
        self,
        image_path: str,
        rois: list[tuple[int, int, int, int]],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        try:
            if output_path is None:
                output_path = self._default_output_path(image_path)

            self._ensure_output_parent(output_path)
            logger.info("Creating anonymized image from ROIs: %s", output_path)

            image = Image.open(image_path).convert("RGB")
            anonymized = self._draw_black_boxes_on_image(image, rois, padding=2)
            anonymized.save(output_path)

            logger.info("Anonymized image saved: %s", output_path)
            return output_path

        except Exception as exc:
            logger.error(
                "Error creating anonymized image from ROIs: %s", exc, exc_info=True
            )
            return None
