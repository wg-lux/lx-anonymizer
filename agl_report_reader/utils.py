import gender_guesser.detector as gender
import random
import re
import hashlib

def replace_umlauts(text):
    """
    Replace German umlauts with their corresponding letter (ä -> a).

    Parameters:
    - text: str
        A string of text.

    Returns:
    - text: str
        The input text with replaced umlauts.
    """
    text = text.replace('ä', 'a')
    text = text.replace('ö', 'o')
    text = text.replace('ü', 'u')
    text = text.replace('Ä', 'A')
    text = text.replace('Ö', 'O')
    text = text.replace('Ü', 'U')
    # text = text.replace('ß', 'ss')

    return text

def pdf_hash(pdf_binary):
    """
    Calculates the SHA256 hash of a PDF file.
    
    Parameters:
    - pdf_binary: bytes
        The binary content of the PDF file.
        
    Returns:
    - hash: str
        The SHA256 hash of the PDF file.
    """
    return hashlib.sha256(pdf_binary).hexdigest()

def remove_titles(name):
        return re.sub(r'(Dr\. med\. |Dr\. |Prof\.)', '', name)

def determine_gender(first_name, detector):
    '''
    The result will be one of unknown (name not found), andy (androgynous), male, female, mostly_male, or mostly_female. \
    The difference between andy and unknown is that the former is found to have the same probability \
    to be male than to be female, while the later means that the name wasn't found in the database.
    '''
    d = gender.Detector()
    country = "germany"

    return d.get_gender(first_name, country)

# get the line starting with PATIENT_INFO_LINE_FLAG
def get_line_by_flag(text, flag):
    for line in text.split("\n"):
        if line.startswith(flag):
            return line
        
def replace_large_numbers(text):
    """
    Replaces all numbers with at least 5 digits in the given text with random numbers of the same length.
    
    Parameters:
    - text: str
        The original text containing numbers.
        
    Returns:
    - new_text: str
        The text with numbers replaced.
    """
    
    # Define a function to generate a random number with 'n' digits
    def random_number(n):
        return ''.join([str(random.randint(0, 9)) for _ in range(n)])
    
    # Find all numbers with at least 5 digits
    numbers_to_replace = re.findall(r'\b\d{5,}\b', text)
    
    # Replace each found number with a random number of the same length
    for number in numbers_to_replace:
        length = len(number)
        random_num = random_number(length)
        text = text.replace(number, random_num)

    return text