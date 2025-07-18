from faker import Faker
from datetime import datetime, timedelta
import random
import re
from .custom_logger import get_logger

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

def anonymize_text(text, report_meta, text_date_format="%d.%m.%Y", 
                   lower_cut_off_flags=None, upper_cut_off_flags=None, 
                   locale="de_DE", first_names=None, last_names=None,
                   apply_cutoffs=False, verbose=False):
    """
    Anonymizes the given text by replacing personal information.
    
    Args:
        text (str): The text to anonymize.
        report_meta (dict): Metadata containing patient and examiner information.
        text_date_format (str): Format for dates in the text.
        lower_cut_off_flags (list): Markers to remove text after a given line.
        upper_cut_off_flags (list): Markers to remove text before a given line.
        locale (str): Locale for generating fake data.
        first_names (list): List of first names to use.
        last_names (list): List of last names to use.
        apply_cutoffs (bool): Whether to apply text cutoffs.
        verbose (bool): Enable verbose logging.
        
    Returns:
        str: Anonymized text.
    """
    logger = get_logger(__name__, verbose=verbose)
    logger.info("Starting text anonymization")
    fake = Faker(locale=locale)
    anonymized_text = text

    # Ersetze Patientennamen, falls Schlüssel in report_meta anders benannt sind, 
    # nehmen wir hier z.B. patient_first_name und patient_last_name:
    if report_meta.get("patient_first_name") and report_meta.get("patient_last_name"):
        first_name = report_meta["patient_first_name"]
        last_name = report_meta["patient_last_name"]
        pseudo_first_name = fake.first_name()
        pseudo_last_name = fake.last_name()
        anonymized_text = anonymized_text.replace(first_name, pseudo_first_name)
        anonymized_text = anonymized_text.replace(last_name, pseudo_last_name)
        logger.debug(f"Replaced patient name: {first_name} {last_name} -> {pseudo_first_name} {pseudo_last_name}")
    
    if report_meta.get("examiner_first_name") and report_meta.get("examiner_last_name"):
        examiner_first_name = report_meta["examiner_first_name"]
        examiner_last_name = report_meta["examiner_last_name"]
        pseudo_examiner_first_name = fake.first_name()
        pseudo_examiner_last_name = fake.last_name()
        anonymized_text = anonymized_text.replace(examiner_first_name, pseudo_examiner_first_name)
        anonymized_text = anonymized_text.replace(examiner_last_name, pseudo_examiner_last_name)
        logger.debug(f"Replaced examiner name: {examiner_first_name} {examiner_last_name} -> {pseudo_examiner_first_name} {pseudo_examiner_last_name}")

    # Verbessertes Cut-Off: Zuerst oberen Teil entfernen, dann unteren.
    if apply_cutoffs:
        logger.info("Applying text cutoffs")
        # Entferne den oberen Teil (alles vor Erleben eines Upper‑Flags)
        if upper_cut_off_flags:
            anonymized_text = cutoff_leading_text(anonymized_text, upper_cut_off_flags)
            logger.debug("Applied upper cutoff")
        # Entferne den unteren Teil (alles nach Erleben eines Lower‑Flags)
        if lower_cut_off_flags:
            anonymized_text = cutoff_trailing_text(anonymized_text, lower_cut_off_flags)
            logger.debug("Applied lower cutoff")
    
    logger.info("Text anonymization complete")
    return anonymized_text

