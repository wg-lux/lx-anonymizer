from faker import Faker
from datetime import datetime, timedelta
import random
import re
from .custom_logger import get_logger

logger = get_logger(__name__)

def replace_umlauts(text):
    """Replace German umlauts with their corresponding letter (ä -> ae)."""
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('Ä', 'AE')
    text = text.replace('Ö', 'OE')
    text = text.replace('Ü', 'UE')
    return text

def replace_large_numbers(text):
    """Replaces all numbers with at least 5 digits with random numbers of the same length."""
    def random_number(n):
        return ''.join([str(random.randint(0, 9)) for _ in range(n)])
    
    numbers_to_replace = re.findall(r'\b\d{5,}\b', text)
    
    for number in numbers_to_replace:
        length = len(number)
        random_num = random_number(length)
        text = text.replace(number, random_num)

    return text

def remove_titles(name):
    """Remove titles like 'Dr.' and 'Dr. med.' from names."""
    return re.sub(r'(Dr\. med\. |Dr\. |Prof\.)', '', name)

def cutoff_leading_text(text, cutoff_flags):
    """
    Remove all text above the first occurrence of any cutoff flag.
    Used primarily for removing letterheads in medical reports.
    """
    if not cutoff_flags:  # If no flags provided, return text as is
        return text
        
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for flag in cutoff_flags:
            if flag in line:
                return '\n'.join(lines[i:])
    return text

def cutoff_trailing_text(text, cutoff_flags):
    """
    Remove all text below the first occurrence of any cutoff flag.
    Used primarily for removing footers in medical reports.
    """
    if not cutoff_flags:  # If no flags provided, return text as is
        return text
        
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for flag in cutoff_flags:
            if flag in line:
                return '\n'.join(lines[:i+1])
    return text

def replace_employee_names(text, first_names, last_names, locale=None):
    """Replace known employee names with fake names."""
    fake = Faker(locale=locale)
    if first_names:
        for first_name in first_names:
            if first_name and first_name in text:
                text = text.replace(first_name, fake.first_name())
    
    if last_names:
        for last_name in last_names:
            if last_name and last_name in text:
                text = text.replace(last_name, fake.last_name())
                
    return text

def anonymize_text(
        text,
        report_meta=None,
        text_date_format='%d.%m.%Y',
        lower_cut_off_flags=None,
        upper_cut_off_flags=None,
        locale=None,
        first_names=None,
        last_names=None,
        apply_cutoffs=False
    ):
    """
    Anonymizes text by replacing sensitive information with fake data.
    
    Parameters:
    - text: Text content to anonymize
    - report_meta: Dictionary with extracted metadata like names, dates, etc.
    - text_date_format: Format for dates in the text
    - lower_cut_off_flags: Flags to identify where to cut off trailing text (for reports)
    - upper_cut_off_flags: Flags to identify where to cut off leading text (for reports)
    - locale: Locale for generating fake data
    - first_names: List of employee first names to anonymize
    - last_names: List of employee last names to anonymize
    - apply_cutoffs: Whether to apply the cutoff functionality (default: False)
    
    Returns:
    - Anonymized text content
    """
    if text is None:
        return None
        
    fake = Faker(locale=locale if locale else 'de_DE')
    
    # Initialize defaults
    report_meta = report_meta or {}
    first_names = first_names or []
    last_names = last_names or []
    lower_cut_off_flags = lower_cut_off_flags or []
    upper_cut_off_flags = upper_cut_off_flags or []
    
    # Loop through each key-value pair in report_meta to replace names and dates
    for key, value in report_meta.items():
        # Skip if value is None or empty
        if not value:
            continue
            
        # Remove titles and replace names
        if 'first_name' in key:
            clean_name = remove_titles(value)
            fake_name = fake.first_name()
            text = text.replace(clean_name, fake_name)
            
        if 'last_name' in key:
            clean_name = remove_titles(value)
            fake_name = fake.last_name()
            text = text.replace(clean_name, fake_name)
        
        # Replace patient's birthdate with a random date in the same year
        if ('birthdate' in key or 'dob' in key):
            try:
                birth_date = datetime.strptime(value, '%Y-%m-%d')
                random_birthdate = datetime(birth_date.year, random.randint(1, 12), random.randint(1, 28))
                formatted_date = random_birthdate.strftime(text_date_format)
                text = text.replace(datetime.strftime(birth_date, text_date_format), formatted_date)
            except (ValueError, TypeError):
                pass
        
        # Replace examination date with a random date in the same month
        if 'examination_date' in key:
            try:
                exam_date = datetime.strptime(value, '%Y-%m-%d')
                random_exam_date = exam_date + timedelta(days=random.randint(-15, 15))
                formatted_date = random_exam_date.strftime(text_date_format)
                text = text.replace(datetime.strftime(exam_date, text_date_format), formatted_date)
            except (ValueError, TypeError):
                pass
    
    # Replace employee names 
    text = replace_employee_names(text, first_names, last_names, locale=locale)
    
    # Replace large numbers (like case numbers, phone numbers, etc.)
    text = replace_large_numbers(text)
    
    # Cut off text based on flags only if apply_cutoffs is True
    if apply_cutoffs:
        if upper_cut_off_flags:
            text = cutoff_leading_text(text, upper_cut_off_flags)
        if lower_cut_off_flags:
            text = cutoff_trailing_text(text, lower_cut_off_flags)
    
    return text

