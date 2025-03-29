from faker import Faker
from datetime import datetime, timedelta
import random
import re
from ..utils import replace_large_numbers
from .redact import cutoff_leading_text, cutoff_trailing_text#

def replace_employee_names(text, first_names, last_names, locale = None):
    fake = Faker(locale=locale)
    for first_name in first_names:
        text = text.replace(first_name, fake.first_name())
    for last_name in last_names:
        text = text.replace(last_name, fake.last_name())

    return text

def anonymize_report(
        text,
        report_meta,
        text_date_format='%d.%m.%Y',
        lower_cut_off_flags = [],
        upper_cut_off_flags = [],
        locale = None,
        first_names = [],
        last_names = []
    ):
    """
    Anonymizes a medical report by replacing real names and dates with fake ones.

    Parameters:
    - text: str
        The original text of the medical report.
    - report_meta: dict
        Dictionary containing metadata of the report, like patient names, birthdate, etc.
    - text_date_format: str
        The date format in the original text (default is '%d.%m.%Y').

    Returns:
    - anonymized_text: str
        The anonymized version of the original text.
    """
    
    fake = Faker(locale=locale)
    
    # Remove titles like 'Dr.' and 'Dr. med.' from names
    def remove_titles(name):
        return re.sub(r'(Dr\. med\. |Dr\. |Prof\.)', '', name)
    
    # ic("Anonymizing report...")
    # ic("Report meta:")
    # _log = pprint(report_meta)
    # ic(_log)
    # _log = pprint(text)
    # ic(_log)
    
    # Loop through each key-value pair in report_meta to replace names and dates
    for key, value in report_meta.items():
        # Remove titles and replace names
        if 'first_name' in key and value:
            clean_name = remove_titles(value)
            fake_name = fake.first_name()
            text = text.replace(clean_name, fake_name)
            
        if 'last_name' in key and value:
            clean_name = remove_titles(value)
            fake_name = fake.last_name()
            text = text.replace(clean_name, fake_name)
        
        # Replace patient's birthdate with a random date in the same year
        if 'dob' in key and value:
            birth_date = datetime.strptime(value, '%Y-%m-%d')
            random_birthdate = datetime(birth_date.year, random.randint(1, 12), random.randint(1, 28))
            formatted_date = random_birthdate.strftime(text_date_format)
            text = text.replace(datetime.strftime(birth_date, text_date_format), formatted_date)
        
        # Replace examination date with a random date in the same month
        if 'examination_date' in key and value:
            exam_date = datetime.strptime(value, '%Y-%m-%d')
            random_exam_date = exam_date + timedelta(days=random.randint(-15, 15))
            formatted_date = random_exam_date.strftime(text_date_format)
            text = text.replace(datetime.strftime(exam_date, text_date_format), formatted_date)

    text = replace_employee_names(text, first_names, last_names, locale = locale)
    text = replace_large_numbers(text)

    
    # ic("Removing leading and trailing text...")
    # ic("Upper cut off flags:")
    # ic(upper_cut_off_flags)
    # ic("Lower cut off flags:")
    # ic(lower_cut_off_flags)
    # ic("Text before cut off:")
    # ic(text)

    # Remove all text above the upper cutoff flag
    text = cutoff_leading_text(text, upper_cut_off_flags)

    # Remove all text below the lower cutoff flag
    text = cutoff_trailing_text(text, lower_cut_off_flags)

    return text