import logging
import re
from typing import Literal, Optional

logger = logging.getLogger(__name__)

class OCRFieldValidator:
    """
    Generalized OCR field validator for examiner, names, and date of birth.
    Provides heuristic validation for plausible real-world values.
    """

    @staticmethod
    def _has_excessive_special_chars(value: str, allowed: str = ".-") -> bool:
        """Return True if value has too many special characters (over 30%)."""
        if not value:
            return True
        special_char_count = sum(1 for c in value if c in allowed)
        ratio = special_char_count / len(value)
        if ratio > 0.3:
            logger.debug(f"Rejected '{value}': too many special chars ({ratio:.2%})")
            return True
        return False

    @staticmethod
    def _is_reasonable_word_structure(value: str) -> bool:
        """Check for normal word structure with at least one valid word."""
        words = value.split()
        valid_words = [
            w for w in words
            if len(w) >= 2 and w.replace("-", "").replace(".", "").isalpha()
        ]
        if not valid_words or len(valid_words) < len(words) / 2:
            logger.debug(
                f"Rejected '{value}': insufficient valid words ({len(valid_words)}/{len(words)})"
            )
            return False
        return True

    @staticmethod
    def validate(
        field_type: Literal["examiner_first_name", "examiner_last_name", "patient_first_name", "patient_last_name", "dob", "casenumber", "center"],
        value: Optional[str]
    ) -> bool:
        """
        Validate OCR-extracted field based on expected type.

        Args:
            field_type: One of 'examiner', 'patient_first_name', 'patient_last_name', 'dob', 'casenumber', 'center'
            value: OCR-extracted string

        Returns:
            bool: True if plausible, False if likely OCR noise
        """
        if not value or not isinstance(value, str):
            return False

        value = value.strip()
        if not (3 <= len(value) <= 50):
            return False

        # --- Common garbage filters ---
        if OCRFieldValidator._has_excessive_special_chars(value):
            return False

        # --- Type-specific validation ---
        if field_type in {"examiner_first_name", "examiner_last_name", "patient_first_name", "patient_last_name"}:
            if not OCRFieldValidator._is_reasonable_word_structure(value):
                return False

            # First/last name specific: usually one or two words, capitalized
            if field_type in {"patient_first_name", "patient_last_name", "examiner_first_name", "examiner_last_name"}:
                if len(value.split()) > 3:
                    logger.debug(f"Rejected {field_type}: too many parts ({value})")
                    return False
                if not re.match(r"^[A-ZÄÖÜ][a-zäöüß\-\. ]+$", value):
                    logger.debug(f"Rejected {field_type}: bad capitalization ({value})")
                    return False

        elif field_type == "patient_dob" or field_type == "examination_date":
            # Date of birth: must look like a date
            # Accepts DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD, etc.
            date_patterns = [
                r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$",
                r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}$"
            ]
            if not any(re.match(p, value) for p in date_patterns):
                logger.debug(f"Rejected dob: not a valid date format ({value})")
                return False

            # Reject dates that are clearly nonsense (e.g., 00.00.0000)
            if re.search(r"00[./-]00[./-]0000", value):
                logger.debug(f"Rejected dob: zero date ({value})")
                return False
        
        elif field_type == "casenumber":
            # Case number: alphanumeric, may include dashes or slashes
            if not re.match(r"^[A-Za-z0-9\-\/]+$", value):
                logger.debug(f"Rejected casenumber: invalid characters ({value})")
                return False

        elif field_type == "center":
            # Center: must be a valid medical center name
            if not OCRFieldValidator._is_reasonable_word_structure(value):
                return False

        return True
