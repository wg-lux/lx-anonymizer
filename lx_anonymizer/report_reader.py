import hashlib
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dateparser
import gender_guesser.detector as gender_detector
import pdfplumber
from faker import Faker
from PIL import Image

from lx_anonymizer.anonymization.anonymizer import Anonymizer
from lx_anonymizer.anonymization.sensitive_region_cropper import \
    SensitiveRegionCropper  # Import the new cropper
from lx_anonymizer.anonymization.text_anonymizer import anonymize_text
from lx_anonymizer.image_processing.pdf_operations import convert_pdf_to_images
from lx_anonymizer.ner.spacy_extractor import (EndoscopeDataExtractor,
                                               ExaminationDataExtractor,
                                               ExaminerDataExtractor,
                                               PatientDataExtractor)
from lx_anonymizer.ner.spacy_ner_fallback import extract_patient_info_from_text
from lx_anonymizer.ocr.ocr import \
    tesseract_full_image_ocr  # , trocr_full_image_ocr_on_boxes # Import OCR fallback
from lx_anonymizer.ocr.ocr_ensemble import \
    ensemble_ocr  # Import the new ensemble OCR
from lx_anonymizer.ollama.ollama_llm_meta_extraction import \
    OllamaOptimizedExtractor
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta
from lx_anonymizer.setup.custom_logger import logger
from lx_anonymizer.setup.private_settings import DEFAULT_SETTINGS
from lx_anonymizer.utils.ollama import ensure_ollama


class ReportReader:
    def __init__(
        self,
        report_root_path: Optional[str] = None,
        locale: str = DEFAULT_SETTINGS["locale"],
        employee_first_names: Optional[List[str]] = None,
        employee_last_names: Optional[List[str]] = None,
        flags: Optional[Dict[Any, Any]] = None,
        text_date_format: str = DEFAULT_SETTINGS["text_date_format"],
    ):
        self.report_root_path = report_root_path

        self.locale = locale
        self.text_date_format = text_date_format
        self.employee_first_names = (
            employee_first_names
            if employee_first_names is not None
            else DEFAULT_SETTINGS["first_names"]
        )
        self.employee_last_names = (
            employee_last_names
            if employee_last_names is not None
            else DEFAULT_SETTINGS["last_names"]
        )
        self.flags = flags if flags is not None else lx_anonymizer/setup/settings.py["flags"]
        self.fake = Faker(locale=locale)
        self.gender_detector = gender_detector.Detector(case_sensitive=True)

        # Initialize extractors
        self.patient_extractor = PatientDataExtractor()
        self.examiner_extractor = ExaminerDataExtractor()
        self.endoscope_extractor = EndoscopeDataExtractor()
        self.examination_extractor = ExaminationDataExtractor()

        # Initialize sensitive region cropper
        self.sensitive_cropper = SensitiveRegionCropper()

        # Initialize Anonymizer
        self.anonymizer = Anonymizer()

        # Initialize Ollama (with graceful degradation)
        self.ollama_proc = None
        self.ollama_extractor = None
        self.ollama_available = False

        # initialize global sensitive meta
        self.sensitive_meta = SensitiveMeta()

        try:
            self.ollama_proc = ensure_ollama()

            # Try to initialize OllamaOptimizedExtractor
            self.ollama_extractor = OllamaOptimizedExtractor()

            # Check if models are available
            if self.ollama_extractor and self.ollama_extractor.current_model:
                self.ollama_available = True
                logger.info("Ollama LLM features enabled for ReportReader")
            else:
                logger.warning(
                    "Ollama models not available, LLM extraction disabled for ReportReader"
                )
                self.ollama_available = False
                self.ollama_extractor = None

        except Exception as e:
            logger.warning(
                f"Ollama/LLM unavailable for ReportReader, will use SpaCy/regex fallback: {e}"
            )
            self.ollama_available = False
            self.ollama_proc = None
            self.ollama_extractor = None

    def process_report(
        self,
        pdf_path=None,
        image_path=None,
        use_ensemble=False,
        verbose=True,
        use_llm_extractor="deepseek",
        text=None,
        create_anonymized_pdf=False,
        anonymized_pdf_output_path=None,
    ) -> Tuple[str, str, dict[Any, Any], Path]:
        """
        Process a report by extracting text, metadata, and creating an anonymized version.
        """

        if text is None:
            if not pdf_path and not image_path:
                raise ValueError(
                    "Either 'pdf_path' 'image_path' or 'text' must be provided."
                )
            if isinstance(pdf_path, str):
                pdf_path = Path(pdf_path)

            if pdf_path:
                if not os.path.exists(pdf_path):
                    logger.error(f"PDF file not found: {pdf_path}")
                    return "", "", {}, Path("")
                text = self.read_pdf(pdf_path)
            elif image_path:
                if not isinstance(image_path, (str, os.PathLike)):
                    logger.error(
                        f"Image path must be a string or PathLike object, got {type(image_path)}: {image_path}"
                    )
                    return "", "", {}, Path("")
                if not os.path.exists(image_path):
                    logger.error(f"Image file not found: {image_path}")
                    return "", "", {}, Path(image_path)
                # If image_path is provided, we assume it's a single image file
                logger.info(f"Reading text from image file: {image_path}")
                try:
                    pil_image = Image.open(image_path)
                    text, _ = tesseract_full_image_ocr(
                        pil_image
                    )  # Use Tesseract OCR on the image
                except Exception as e:
                    logger.error(f"Error reading image {image_path}: {e}")
                    return "", "", {}, Path(image_path)

        # --- OCR Fallback ---
        if (
            not text or len(text.strip()) < 50
        ):  # Trigger OCR if text is empty or very short
            try:
                assert isinstance(text, str)
                logger.info(
                    f"Short/No text detected by pdfplumber ({len(text.strip())} chars), applying OCR fallback."
                )

                if pdf_path:
                    if not isinstance(pdf_path, (str, os.PathLike, Path)):
                        logger.error(
                            f"Cannot apply OCR: PDF path is not valid: {pdf_path}"
                        )
                        return "", "", {}, Path(str(pdf_path or ""))

                    logger.info(f"Converting PDF to images for OCR: {pdf_path}")
                    try:
                        images_from_pdf = convert_pdf_to_images(Path(pdf_path))
                    except Exception as e:
                        logger.error(f"Failed to convert PDF to images: {e}")
                        return "", "", {}, Path(pdf_path)
                elif image_path:
                    if not isinstance(image_path, (str, os.PathLike, Path)):
                        logger.error(
                            f"Cannot apply OCR: Image path is not valid: {image_path}"
                        )
                        return "", "", {}, Path(image_path)
                    try:
                        images_from_pdf = [Image.open(str(image_path))]
                    except Exception as e:
                        logger.error(f"Failed to open image file: {e}")
                        return "", "", {}, Path(image_path)
                else:
                    logger.error("No valid path provided for OCR processing")
                    return "", "", {}, Path("")

                ocr_text = ""

                for idx, pil_image in enumerate(images_from_pdf):
                    if pil_image is None:
                        logger.error(f"Page {idx + 1} image is None, skipping OCR")
                        continue

                    logger.info(f"Processing page {idx + 1} with OCR...")
                    ocr_part = ""
                    if use_ensemble:
                        logger.info("Using ensemble OCR approach")
                        try:
                            ocr_part = ensemble_ocr(pil_image)
                        except Exception as e:
                            logger.error(f"Ensemble OCR failed on page {idx + 1}: {e}")
                            try:
                                logger.info(
                                    "Falling back to Tesseract OCR after ensemble failure."
                                )
                                text_part, _ = tesseract_full_image_ocr(pil_image)
                                ocr_part = text_part
                            except Exception as te:
                                logger.error(
                                    f"Tesseract fallback also failed on page {idx + 1}: {te}"
                                )
                                ocr_part = ""
                    else:
                        try:
                            text_part, _ = tesseract_full_image_ocr(pil_image)
                            ocr_part = text_part
                            logger.info(f"Tesseract OCR successful for page {idx + 1}")
                        except Exception as e:
                            logger.error(f"Tesseract OCR failed on page {idx + 1}: {e}")
                            ocr_part = ""

                    ocr_text += (
                        " " + ocr_part if ocr_part else ""
                    )  # Append only if OCR succeeded

                text = ocr_text.strip()
                logger.info(
                    f"OCR fallback finished. Total text length: {len(text)}. Preview: {text[:200]}..."
                )

                # Apply correction using Ollama (Keep this if you want to correct OCR text)
                if (
                    text and len(text.strip()) > 10
                ):  # Only correct if OCR produced something meaningful
                    logger.info("Applying LLM correction to OCR text via Ollama")
                    try:
                        from lx_anonymizer.ollama.ollama_service import \
                            ollama_service  # Import locally if needed

                        corrected_text = ollama_service.correct_ocr_text_in_chunks(text)

                        if (
                            corrected_text
                            and corrected_text != text
                            and len(corrected_text) > 0.5 * len(text)
                        ):
                            logger.info("OCR text successfully corrected by Ollama.")
                            text = corrected_text
                        elif corrected_text == text:
                            logger.info("Ollama correction resulted in the same text.")
                        else:
                            logger.warning(
                                "Ollama OCR correction failed or produced poor result, using original OCR text."
                            )
                    except Exception as e:
                        logger.warning(f"Error using Ollama for correction: {e}")

                if not text or len(text.strip()) < 10:
                    logger.error(
                        "OCR fallback produced very short/no text, cannot proceed with metadata extraction."
                    )
                    original_text_from_pdf = self.read_pdf(
                        pdf_path
                    )  # Re-read original for context
                    return (
                        original_text_from_pdf,
                        original_text_from_pdf,
                        {},
                        Path(str(pdf_path or "")),
                    )

            except Exception as e:
                logger.error(f"OCR fallback process failed entirely: {e}")
                original_text_from_pdf = self.read_pdf(
                    pdf_path
                )  # Re-read original for context
                return (
                    original_text_from_pdf,
                    original_text_from_pdf,
                    {},
                    Path(str(pdf_path or "")),
                )

        # --- Metadata Extraction ---
        report_meta = {}
        if text and len(text.strip()) >= 10:  # Proceed only if we have some text
            if use_llm_extractor:
                logger.info(f"Using specified LLM extractor: {use_llm_extractor}")
                if use_llm_extractor == "deepseek":
                    report_meta = self.extract_report_meta_deepseek(text)
                elif use_llm_extractor == "medllama":
                    report_meta = self.extract_report_meta_medllama(text)
                elif use_llm_extractor == "llama3":
                    report_meta = self.extract_report_meta_llama3(text)
                else:
                    logger.warning(
                        f"Unknown LLM extractor specified: {use_llm_extractor}. Falling back to default."
                    )
                    report_meta = self.extract_report_meta(
                        text, pdf_path=None
                    )  # Default SpaCy/Regex

                # FIX: Ensure report_meta is a valid dict before updating sensitive_meta
                if report_meta:
                    self.sensitive_meta.safe_update(report_meta)
                    report_meta = self.sensitive_meta.to_dict()
                else:
                    logger.warning(
                        f"LLM extractor '{use_llm_extractor}' failed. Falling back to default SpaCy/Regex extraction."
                    )
                    report_meta = self.extract_report_meta(text, pdf_path)
            else:
                logger.info("Using default SpaCy/Regex metadata extraction.")
                report_meta = self.extract_report_meta(text, pdf_path)
                self.sensitive_meta.safe_update(report_meta)
                report_meta = self.sensitive_meta.to_dict()
        else:
            logger.warning(
                "Skipping metadata extraction due to insufficient text content."
            )
            report_meta = {
                "pdf_hash": self.pdf_hash(open(str(pdf_path), "rb").read())
                if pdf_path and os.path.exists(str(pdf_path))
                else None
            }

        self.sensitive_meta.safe_update(report_meta)

        # --- Anonymization ---
        anonymized_text = self.anonymize_report(text=text, report_meta=report_meta)

        # --- Create Anonymized PDF (if requested) ---
        anonymized_pdf_path = None

        if create_anonymized_pdf and pdf_path:
            try:
                logger.info(
                    "Creating anonymized PDF with blackened sensitive regions..."
                )
                anonymized_pdf_path = self.anonymizer.create_anonymized_pdf(
                    pdf_path=str(pdf_path),
                    output_path=anonymized_pdf_output_path,
                    report_meta=report_meta,
                )
                if anonymized_pdf_path:
                    report_meta["anonymized_pdf_path"] = anonymized_pdf_path
                    logger.info(f"Anonymized PDF created: {anonymized_pdf_path}")
                else:
                    logger.warning("Failed to create anonymized PDF")
            except Exception as e:
                logger.error(f"Error creating anonymized PDF: {e}")
                report_meta["anonymized_pdf_error"] = str(e)
        try:
            assert isinstance(text, str)
            report_meta["text"] = text
        except AssertionError:
            report_meta["text"] = "Unknown"
        try:
            assert isinstance(anonymized_text, str)
            report_meta["anonymized_text"] = anonymized_text
        except AssertionError:
            report_meta["anonymized_text"] = "Unknown"

        sensitive_meta = dict(
            file_path=str(pdf_path) if pdf_path else None,
            patient_first_name=report_meta.get("patient_first_name")
            or report_meta.get("first_name"),
            patient_last_name=report_meta.get("patient_last_name")
            or report_meta.get("last_name"),
            patient_dob=report_meta.get("patient_dob") or report_meta.get("birth_date"),
            casenumber=report_meta.get("casenumber") or report_meta.get("casenumber"),
            patient_gender_name=report_meta.get("patient_gender_name")
            or report_meta.get("gender"),
            examination_date=report_meta.get("examination_date"),
            examination_time=report_meta.get("examination_time"),
            examiner_first_name=report_meta.get("examiner_first_name")
            or report_meta.get("doctor_first_name"),
            examiner_last_name=report_meta.get("examiner_last_name")
            or report_meta.get("doctor_last_name"),
            center=report_meta.get("center") or report_meta.get("hospital"),
            text=text,
            anonymized_text=anonymized_text,
        )
        self.sensitive_meta.safe_update(sensitive_meta)

        # Ensure return tuple size is 4
        return (
            text,
            anonymized_text,
            self.sensitive_meta.to_dict(),
            Path(str(anonymized_pdf_path)) if anonymized_pdf_path else Path("None"),
        )

    def read_pdf(self, pdf_path):
        """Read pdf file using pdfplumber and return the raw text content."""
        if pdf_path is None:
            logger.error("PDF path is None, cannot read PDF")
            return ""

        try:
            pdf_path = str(pdf_path)
        except Exception:
            logger.error(f"Cannot convert pdf_path to string: {pdf_path}")
            return ""

        # Disable verbose pdfminer logging
        logging.getLogger("pdfminer").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.psparser").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfdocument").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfinterp").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfpage").setLevel(logging.WARNING)

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def extract_report_meta(self, text, pdf_path) -> Dict[str, Any]:
        """Extract metadata from report text using the spacy extractor classes."""
        report_meta = {}

        logger.debug(f"Full text extracted from PDF: {text[:500]}...")
        lines = text.split("\n") if text else []

        patient_info = None
        # Option 1: Try on whole text with Deepseek
        if text:
            # Note: We don't call extract_report_meta_deepseek here directly to prevent recursion if called from within process_report fallback
            # But the logic suggests if we are here, we might have already tried LLM or opted out.
            # Just use patient_extractor (SpaCy)
            patient_info = self.patient_extractor(text)
            logger.debug(f"Patient extractor result on full text: {patient_info}")
        else:
            logger.debug("Skipping extraction on empty text.")

        is_valid_info = patient_info and (
            patient_info.get("patient_first_name") is not None
            or patient_info.get("patient_last_name") is not None
        )

        # Option 2: If full text failed, try line by line with SpaCy
        if not is_valid_info and lines:
            logger.debug("Extractor failed on full text, trying line by line.")
            for line in lines:
                if re.search(r"pat(?:ient|ientin|\.|iont|bien)", line, re.IGNORECASE):
                    patient_info_line = self.patient_extractor(line)
                    is_valid_line_info = patient_info_line and (
                        patient_info_line.get("patient_first_name") is not None
                        or patient_info_line.get("patient_last_name") is not None
                    )
                    if is_valid_line_info:
                        patient_info = patient_info_line
                        is_valid_info = True
                        break

        if not is_valid_info and text:
            logger.debug("SpaCy extractor failed, using regex fallback extraction")
            fallback_info = extract_patient_info_from_text(text)
            if (
                fallback_info.get("patient_first_name") != "Unknown"
                or fallback_info.get("patient_last_name") != "Unknown"
            ):
                # Clean up "Unknown" strings
                for k, v in fallback_info.items():
                    if v == "Unknown":
                        fallback_info[k] = None
                patient_info = fallback_info
                is_valid_info = True
            else:
                if not patient_info:
                    patient_info = PatientDataExtractor._blank()

        if not patient_info:
            patient_info = PatientDataExtractor._blank()

        # Date Parsing
        dob_value = patient_info.get("patient_dob")
        parsed_dob = None
        if isinstance(dob_value, str):
            parsed_dob = dateparser.parse(
                dob_value, languages=["de"], settings={"DATE_ORDER": "DMY"}
            )
            if parsed_dob:
                parsed_dob = parsed_dob.date()
        elif isinstance(dob_value, (datetime, date)):
            parsed_dob = dob_value if isinstance(dob_value, date) else dob_value.date()

        final_patient_info = {
            "patient_first_name": patient_info.get("patient_first_name"),
            "patient_last_name": patient_info.get("patient_last_name"),
            "patient_dob": parsed_dob,
            "casenumber": patient_info.get("casenumber"),
            "patient_gender_name": patient_info.get("patient_gender_name"),
        }
        self.sensitive_meta.safe_update(final_patient_info)
        report_meta = self.sensitive_meta.to_dict()

        # Extract other information
        if lines:
            for line in lines:
                if re.search(r"unters\w*\s*arzt", line, re.IGNORECASE):
                    examiner_info = self.examiner_extractor.extract_examiner_info(line)
                    if examiner_info:
                        report_meta.update(examiner_info)

                if re.search(r"unters\.:|u-datum:|eingang\s*am:", line, re.IGNORECASE):
                    examination_info = (
                        self.examination_extractor.extract_examination_info(line)
                    )
                    if examination_info and examination_info.get("examination_date"):
                        report_meta.update(examination_info)
                        # Break not strict here, assuming multiple lines

                if self.flags.get("endoscope_info_line", "").lower() in line.lower():
                    endoscope_info = self.endoscope_extractor.extract_endoscope_info(
                        line
                    )
                    if endoscope_info:
                        report_meta.update(endoscope_info)

        self.sensitive_meta.safe_update(report_meta)
        report_meta = self.sensitive_meta.to_dict()

        # PDF Hash
        try:
            if pdf_path and os.path.exists(str(pdf_path)):
                with open(str(pdf_path), "rb") as f:
                    pdf_bytes = f.read()
                    report_meta["pdf_hash"] = self.pdf_hash(pdf_bytes)
            else:
                report_meta["pdf_hash"] = None
        except Exception as e:
            logger.error(f"Could not calculate PDF hash: {e}")
            report_meta["pdf_hash"] = None

        return report_meta

    def extract_report_meta_deepseek(self, text):
        """Extract metadata using DeepSeek via Ollama structured output."""
        if not self.ollama_available or not self.ollama_extractor:
            logger.warning(
                "Ollama not available for DeepSeek extraction, returning empty dict."
            )
            return {}

        logger.info(
            "Attempting metadata extraction with DeepSeek (Ollama Structured Output)"
        )
        try:
            meta_obj = self.ollama_extractor.extract_metadata(text)

            # FIX: safe_update returns None. Must call .to_dict() after update.
            if meta_obj:
                self.sensitive_meta.safe_update(meta_obj)
                meta = self.sensitive_meta.to_dict()
                logger.info("DeepSeek Ollama extraction successful.")
            else:
                meta = {}
                logger.warning(
                    "DeepSeek Ollama extraction failed, returning empty dict."
                )

            return meta
        except Exception as e:
            logger.warning(f"DeepSeek Ollama extraction error: {e}")
            return {}

    def extract_report_meta_medllama(self, text):
        if not self.ollama_available or not self.ollama_extractor:
            return {}
        try:
            meta_obj = self.ollama_extractor.extract_metadata(text)
            if meta_obj:
                self.sensitive_meta.safe_update(meta_obj)
                return self.sensitive_meta.to_dict()
            return {}
        except Exception as e:
            logger.warning(f"MedLLaMA Ollama extraction error: {e}")
            return {}

    def extract_report_meta_llama3(self, text):
        if not self.ollama_available or not self.ollama_extractor:
            return {}
        try:
            meta_obj = self.ollama_extractor.extract_metadata(text)
            if meta_obj:
                self.sensitive_meta.safe_update(meta_obj)
                return self.sensitive_meta.to_dict()
            return {}
        except Exception as e:
            logger.warning(f"Llama3 Ollama extraction error: {e}")
            return {}

    def anonymize_report(self, text, report_meta):
        """Anonymize the report text using the extracted metadata."""
        anonymized_text = anonymize_text(
            text=text,
            report_meta=report_meta,
            text_date_format=self.text_date_format,
            lower_cut_off_flags=self.flags["cut_off_below"],
            upper_cut_off_flags=self.flags["cut_off_above"],
            locale=self.locale,
            first_names=self.employee_first_names,
            last_names=self.employee_last_names,
            apply_cutoffs=True,
        )
        return anonymized_text

    def pdf_hash(self, pdf_binary):
        return hashlib.sha256(pdf_binary).hexdigest()

    def process_report_with_cropping(
        self,
        pdf_path=None,
        image_path=None,
        use_ensemble=False,
        verbose=True,
        use_llm_extractor="deepseek",
        text=None,
        crop_output_dir=None,
        crop_sensitive_regions=True,
        anonymization_output_dir=None,
    ):
        """Extended version of process_report with optional cropping."""
        original_text, anonymized_text, report_meta, _ = self.process_report(
            pdf_path=pdf_path,
            image_path=image_path,
            use_ensemble=use_ensemble,
            verbose=verbose,
            use_llm_extractor=use_llm_extractor,
            text=text,
        )

        cropped_regions_info = {}
        anonymized_pdf_path = None

        if not crop_output_dir:
            crop_output_dir = Path(os.getcwd()).parent / "pdfs" / "cropped_regions"

        if not anonymization_output_dir:
            anonymization_output_dir = Path(os.getcwd()).parent / "pdfs" / "anonymized"

        if crop_sensitive_regions and crop_output_dir and pdf_path:
            try:
                logger.info("Beginne Cropping sensitiver Regionen...")
                cropped_regions_info = self.sensitive_cropper.crop_sensitive_regions(
                    pdf_path=pdf_path, output_dir=str(crop_output_dir)
                )
            except Exception as e:
                logger.error(
                    f"Fehler beim initialien Aufruf der Funktion zum Cropping: {e}"
                )
                cropped_regions_info = {}

            if cropped_regions_info:
                out_dir = (
                    Path(anonymization_output_dir)
                    if anonymization_output_dir
                    else Path(pdf_path).parent
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                anonymized_pdf_path = out_dir / (Path(pdf_path).stem + ".pdf")
                try:
                    self.sensitive_cropper.create_anonymized_pdf_with_crops(
                        pdf_path=pdf_path,
                        crop_output_dir=str(crop_output_dir),
                        anonymized_pdf_path=str(anonymized_pdf_path),
                    )
                    report_meta["anonymized_pdf_path"] = str(anonymized_pdf_path)
                except Exception as pdf_error:
                    logger.warning(
                        f"Konnte anonymisiertes PDF nicht erstellen: {pdf_error}"
                    )
                    report_meta["anonymized_pdf_error"] = str(pdf_error)
                    anonymized_pdf_path = None

            report_meta["cropped_regions"] = cropped_regions_info
            report_meta["cropping_enabled"] = True
            report_meta["total_cropped_regions"] = sum(
                len(crops) for crops in cropped_regions_info.values()
            )
        else:
            report_meta["cropping_enabled"] = False

        return (
            original_text,
            anonymized_text,
            report_meta,
            cropped_regions_info,
            anonymized_pdf_path,
        )

    def create_visualization_report(
        self, pdf_path, output_dir, visualize_all_pages=False
    ):
        """Create visualization report for sensitive regions."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_name = Path(pdf_path).stem
        images = convert_pdf_to_images(pdf_path)
        visualization_files = []

        pages_to_process = range(len(images)) if visualize_all_pages else [0]

        for page_num in pages_to_process:
            if page_num >= len(images):
                continue

            image = images[page_num]
            full_text, word_boxes = tesseract_full_image_ocr(image)

            vis_filename = f"{pdf_name}_page_{page_num + 1}_analysis.png"
            vis_path = output_dir / vis_filename

            self.sensitive_cropper.visualize_sensitive_regions(
                image, word_boxes, str(vis_path)
            )
            visualization_files.append(str(vis_path))

        return visualization_files

        return visualization_files
