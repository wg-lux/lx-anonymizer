from typing import Optional, Dict, Any
from pathlib import Path

import fitz

from .east_text_detection import east_text_detection
from .ocr_tesserocr import tesseract_on_boxes_fast
from .pdf_operations import convert_pdf_to_images
from .custom_logger import get_logger
from .sensitive_region_cropper import SensitiveRegionCropper


logger = get_logger(__name__)

class Anonymizer:
    def __init__(self) -> None:
        self.sensitive_cropper = SensitiveRegionCropper()
        pass
    
    def create_anonymized_pdf(self, pdf_path: str, output_path: Optional[str] = None, report_meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create an anonymized PDF with sensitive regions blackened out.
        
        Uses EAST text detection for fast region identification and TesseOCR for 
        accurate text recognition and sensitivity classification.

        Args:
            pdf_path: Path to the original PDF
            output_path: Path for the anonymized PDF (optional, auto-generated if not provided)
            report_meta: Metadata from extraction (optional, will be extracted if not provided)

        Returns:
            Path to the anonymized PDF, or None if creation failed
        """
        try:
            if not output_path:
                output_path = str(Path(pdf_path).with_stem(Path(pdf_path).stem + "_anonymized"))

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Creating anonymized PDF with EAST + TesseOCR: {output_path}")

            # Open the original PDF
            doc = fitz.open(pdf_path)

            # Convert PDF to images for analysis
            images = convert_pdf_to_images(pdf_path)

            # Process each page
            for page_num, image in enumerate(images):
                page = doc[page_num]

                # Save temporary image for EAST detection
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    temp_image_path = tmp_file.name
                    image.save(temp_image_path)

                try:
                    # Step 1: Use EAST for fast text region detection
                    logger.debug(f"Running EAST text detection on page {page_num + 1}")
                    text_boxes, confidences_json = east_text_detection(
                        temp_image_path,
                        min_confidence=0.5,
                        width=640,  # Higher resolution for better detection
                        height=640
                    )

                    if not text_boxes:
                        logger.info(f"No text regions detected on page {page_num + 1}")
                        continue

                    logger.info(f"EAST detected {len(text_boxes)} text regions on page {page_num + 1}")

                    # Step 2: Use TesseOCR for fast and accurate text extraction
                    logger.debug(f"Running TesseOCR on {len(text_boxes)} boxes")
                    text_with_boxes, ocr_confidences = tesseract_on_boxes_fast(
                        image,  # Use PIL image directly
                        text_boxes,
                        language="deu+eng"
                    )

                    # Step 3: Detect sensitive regions based on OCR results
                    sensitive_regions = self.sensitive_cropper.detect_sensitive_regions(
                        image, 
                        text_with_boxes
                    )

                    if sensitive_regions:
                        logger.info(f"Blackening {len(sensitive_regions)} sensitive regions on page {page_num + 1}")

                        # Convert pixel coordinates to PDF coordinates
                        page_rect = page.rect
                        page_height = page_rect.height
                        page_width = page_rect.width

                        img_width, img_height = image.size

                        # Scaling factors
                        scale_x = page_width / img_width
                        scale_y = page_height / img_height

                        for x1, y1, x2, y2 in sensitive_regions:
                            # Convert image coordinates to PDF coordinates
                            pdf_x1 = x1 * scale_x
                            pdf_y1 = page_height - (y2 * scale_y)  # Invert Y-axis
                            pdf_x2 = x2 * scale_x
                            pdf_y2 = page_height - (y1 * scale_y)  # Invert Y-axis

                            # Create black rectangle with slight padding for better coverage
                            padding = 2  # PDF points
                            rect = fitz.Rect(
                                pdf_x1 - padding, 
                                pdf_y1 - padding, 
                                pdf_x2 + padding, 
                                pdf_y2 + padding
                            )

                            # Add black rectangle to cover sensitive area
                            page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))

                            logger.debug(f"Blackened: ({pdf_x1:.1f}, {pdf_y1:.1f}, {pdf_x2:.1f}, {pdf_y2:.1f})")
                    else:
                        logger.info(f"No sensitive regions detected on page {page_num + 1}")

                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(temp_image_path)
                    except Exception:
                        pass

            # Save the anonymized PDF
            doc.save(output_path)
            doc.close()

            logger.info(f"Anonymized PDF saved: {output_path}")
            return output_path

        except ImportError as e:
            logger.error(f"Required module not installed. Cannot create anonymized PDF: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating anonymized PDF: {e}", exc_info=True)
            return None
        
    def create_anonymized_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        report_meta: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create an anonymized image with sensitive regions blackened out.

        Uses EAST text detection for fast region identification and TesseOCR for
        accurate text recognition and sensitivity classification.

        Args:
            image_path: Path to the original image.
            output_path: Path for the anonymized image (optional, auto-generated if not provided).
            report_meta: Metadata from extraction (optional, will be extracted if not provided).

        Returns:
            Path to the anonymized image, or None if creation failed.
        """
        try:
            import tempfile
            from PIL import Image, ImageDraw
            import os

            if not output_path:
                output_path = str(Path(image_path).with_stem(Path(image_path).stem + "_anonymized"))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Creating anonymized image with EAST + TesseOCR: {output_path}")

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Save temporary file for EAST model (expects file path)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                temp_image_path = tmp_file.name
                image.save(temp_image_path)

            try:
                # Step 1: EAST text detection
                logger.debug("Running EAST text detection on image")
                text_boxes, confidences_json = east_text_detection(
                    temp_image_path,
                    min_confidence=0.5,
                    width=640,
                    height=640
                )

                if not text_boxes:
                    logger.info("No text regions detected.")
                    return None

                logger.info(f"EAST detected {len(text_boxes)} text regions.")

                # Step 2: OCR within detected boxes
                logger.debug(f"Running TesseOCR on {len(text_boxes)} boxes")
                text_with_boxes, ocr_confidences = tesseract_on_boxes_fast(
                    image,
                    text_boxes,
                    language="deu+eng"
                )

                # Step 3: Sensitive region detection
                sensitive_regions = self.sensitive_cropper.detect_sensitive_regions(
                    image,
                    text_with_boxes
                )

                if not sensitive_regions:
                    logger.info("No sensitive regions detected. Copying image unchanged.")
                    image.save(output_path)
                    return output_path

                logger.info(f"Blackening {len(sensitive_regions)} sensitive regions.")

                # Draw black rectangles over sensitive areas
                draw = ImageDraw.Draw(image)
                for x1, y1, x2, y2 in sensitive_regions:
                    draw.rectangle([(x1 - 2, y1 - 2), (x2 + 2, y2 + 2)], fill="black")
                    logger.debug(f"Blackened region: ({x1}, {y1}, {x2}, {y2})")

                image.save(output_path)
                logger.info(f"Anonymized image saved: {output_path}")
                return output_path

            finally:
                try:
                    os.unlink(temp_image_path)
                except Exception:
                    pass

        except ImportError as e:
            logger.error(f"Required module not installed. Cannot create anonymized image: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating anonymized image: {e}", exc_info=True)
            return None
        
    def create_anonymized_pdf_from_rois(
        self,
        pdf_path: str,
        rois_per_page: Dict[int, list[tuple[int, int, int, int]]],
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create an anonymized PDF by blackening given ROIs.

        Args:
            pdf_path: Path to the original PDF.
            rois_per_page: Dict mapping page index -> list of (x1, y1, x2, y2) regions (in image pixels).
            output_path: Path for anonymized PDF (optional, auto-generated if not provided).

        Returns:
            Path to the anonymized PDF, or None on failure.
        """
        try:
            if not output_path:
                output_path = str(Path(pdf_path).with_stem(Path(pdf_path).stem + "_anonymized"))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating anonymized PDF from ROIs: {output_path}")

            # Open PDF and its raster images (for scaling)
            doc = fitz.open(pdf_path)
            images = convert_pdf_to_images(pdf_path)

            for page_num, image in enumerate(images):
                page = doc[page_num]
                rois = rois_per_page.get(page_num, [])
                if not rois:
                    continue

                logger.info(f"Blackening {len(rois)} regions on page {page_num + 1}")
                page_rect = page.rect
                page_height, page_width = page_rect.height, page_rect.width
                img_width, img_height = image.size

                scale_x = page_width / img_width
                scale_y = page_height / img_height

                for (x1, y1, x2, y2) in rois:
                    # Convert image to PDF coordinates (invert Y)
                    pdf_x1 = x1 * scale_x
                    pdf_y1 = page_height - (y2 * scale_y)
                    pdf_x2 = x2 * scale_x
                    pdf_y2 = page_height - (y1 * scale_y)
                    rect = fitz.Rect(pdf_x1 - 2, pdf_y1 - 2, pdf_x2 + 2, pdf_y2 + 2)
                    page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))

            doc.save(output_path)
            doc.close()
            logger.info(f"Anonymized PDF saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating anonymized PDF from ROIs: {e}", exc_info=True)
            return None
        
    def create_anonymized_image_from_rois(
        self,
        image_path: str,
        rois: list[tuple[int, int, int, int]],
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create an anonymized image by blackening given ROIs.

        Args:
            image_path: Path to the original image.
            rois: List of (x1, y1, x2, y2) coordinates in pixel space.
            output_path: Path for anonymized image (optional, auto-generated if not provided).

        Returns:
            Path to the anonymized image, or None on failure.
        """
        try:
            from PIL import Image, ImageDraw

            if not output_path:
                output_path = str(Path(image_path).with_stem(Path(image_path).stem + "_anonymized"))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating anonymized image from ROIs: {output_path}")

            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            if not rois:
                logger.info("No ROIs provided — saving unmodified image.")
                image.save(output_path)
                return output_path

            for (x1, y1, x2, y2) in rois:
                draw.rectangle([(x1 - 2, y1 - 2), (x2 + 2, y2 + 2)], fill="black")
                logger.debug(f"Blackened ROI: ({x1}, {y1}, {x2}, {y2})")

            image.save(output_path)
            logger.info(f"Anonymized image saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating anonymized image from ROIs: {e}", exc_info=True)
            return None


