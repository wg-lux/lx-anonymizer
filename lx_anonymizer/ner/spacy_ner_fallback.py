from datetime import datetime

from lx_anonymizer.regex_patterns import (
    FALLBACK_PATIENT_FULL_RE,
    FALLBACK_PATIENT_NAME_RE,
    FALLBACK_PATIENT_TITLED_NAME_RE,
)


def extract_patient_info_from_text(text):
    # Dictionary to store extracted information
    info = {
        "first_name": "Unknown",
        "last_name": "Unknown",
        "dob": None,
        "gender": "Unknown",
        "casenumber": None,
    }

    """
    A more robust patient information extraction using advanced regex patterns.
    This is a fallback method when the SpaCy extractors fail.
    
    Args:
        text (str): The full text from which to extract patient information
        
    Returns:
        dict: Dictionary with patient information or None if no match is found
    """

    # Try multiple patterns to find patient information

    # Pattern 1: Patient info with birth date and case number
    match1 = FALLBACK_PATIENT_FULL_RE.search(text)

    if match1:
        info["last_name"] = match1.group(1).strip()
        info["first_name"] = match1.group(2).strip()

        # Convert date format
        try:
            birth_date = datetime.strptime(match1.group(3), "%d.%m.%Y").strftime(
                "%Y-%m-%d"
            )
            info["dob"] = birth_date
        except (ValueError, TypeError):
            pass

        # Extract case number if available
        if match1.group(4):
            info["casenumber"] = match1.group(4).strip()

        # Determine gender based on context
        # Check for gender indicators near the extracted name
        context_window = text[
            max(0, text.find(info["first_name"]) - 30) : min(
                len(text),
                text.find(info["first_name"])
                + len(info["first_name"])
                + 30,
            )
        ]

        if "Patientin" in context_window:
            info["gender"] = "female"
        elif "Patient" in context_window and "Patientin" not in context_window:
            info["gender"] = "male"

        return info

    # Pattern 2: Simple patient name only
    match2 = FALLBACK_PATIENT_NAME_RE.search(text)

    if match2:
        info["last_name"] = match2.group(1).strip()
        info["first_name"] = match2.group(2).strip()
        return info

    # Pattern 3: Full name in one group with title
    match3 = FALLBACK_PATIENT_TITLED_NAME_RE.search(text)

    if match3:
        full_name = match3.group(1).strip()
        parts = full_name.split()

        if len(parts) >= 2:
            # Assume first part is first name, rest is last name
            info["first_name"] = parts[0]
            info["last_name"] = " ".join(parts[1:])
            return info

    return info
