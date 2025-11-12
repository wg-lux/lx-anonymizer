#!/usr/bin/env python3
import csv
import re
from pathlib import Path
from typing import List, Dict
import difflib

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# Predefined field list
STANDARD_FIELDS = {
    "id", "casenumber", "patient_first_name", "patient_last_name", 
    "patient_dob", "examination_date", "examination_time", "center_name", 
    "patient_gender_name", "endoscope_type", "endoscope_sn", 
    "examiner_first_name", "examiner_last_name", "text", "anonymized_text", 
    "external_id", "external_id_origin"
}

# Directory to apply changes to (your project directory)
ROOT_DIR = Path(__file__).parent

# CSV file with field usages
CSV_FILE = ROOT_DIR / "field_usages.csv"

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def load_field_usages(csv_path: Path) -> List[Dict[str, str]]:
    """Load field usages from the CSV into a list of dictionaries."""
    usages = []
    with csv_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            usages.append(row)
    return usages

def find_closest_match(field: str, standard_fields: set) -> str:
    """Find the closest match for a field in the standard list using difflib."""
    matches = difflib.get_close_matches(field, standard_fields, n=1, cutoff=0.8)
    return matches[0] if matches else field

def normalize_code_line(line: str, field: str, normalized_field: str) -> str:
    """Normalize the code line by replacing the field with the normalized field."""
    # Replace variable/field occurrences in the line (use word boundaries to avoid partial matches)
    return re.sub(rf"\b{re.escape(field)}\b", normalized_field, line)

def add_missing_name_counterparts(line: str) -> str:
    """Add missing first or last name counterpart if one is found."""
    first_name_pattern = r"\bpatient_first_name\b"
    last_name_pattern = r"\bpatient_last_name\b"
    
    # Check if first name is present but last name is missing
    if re.search(first_name_pattern, line) and not re.search(last_name_pattern, line):
        line = re.sub(first_name_pattern, "patient_first_name, patient_last_name", line)
    
    # Check if last name is present but first name is missing
    if re.search(last_name_pattern, line) and not re.search(first_name_pattern, line):
        line = re.sub(last_name_pattern, "patient_first_name, patient_last_name", line)
    
    return line

def normalize_fields_in_files(usages: List[Dict[str, str]], root_dir: Path):
    """Normalize all fields in the code files based on CSV findings and add missing counterparts."""
    for usage in usages:
        file_path = Path(usage["file"])
        line_number = int(usage["line"])
        keyword = usage["keyword"]
        context = usage["context"]
        code_line = usage["code"]
        
        # Find the closest match in the standard fields list
        normalized_field = find_closest_match(keyword, STANDARD_FIELDS)
        
        # Normalize the line by replacing the field
        if normalized_field != keyword:
            print(f"Normalizing: {keyword} → {normalized_field} in {file_path}:{line_number}")
            # Open the file and update the specific line
            with file_path.open("r", encoding="utf-8") as file:
                lines = file.readlines()
            
            # Apply missing name counterparts before normalization
            lines[line_number - 1] = add_missing_name_counterparts(lines[line_number - 1])

            # Normalize the field and write back the updated line
            lines[line_number - 1] = normalize_code_line(lines[line_number - 1], keyword, normalized_field)

            # Write the changes back to the file
            with file_path.open("w", encoding="utf-8") as file:
                file.writelines(lines)

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Normalizing field usages and adding missing name counterparts...")

    # Load field usages from CSV
    field_usages = load_field_usages(CSV_FILE)

    # Normalize fields across code files and add missing counterparts
    normalize_fields_in_files(field_usages, ROOT_DIR)

    print("\n✅ Normalization complete.")
