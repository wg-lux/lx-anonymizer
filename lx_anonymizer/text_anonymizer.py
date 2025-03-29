from flair_NER import flair_NER_German
from custom_logger import get_logger
from names_generator import get_random_name

logger = get_logger(__name__)

def anonymize_text(text):
    '''
    Anonymize Text
    This function takes a text as input and anonymizes it by replacing all person names with "PERSON".
    input: text (str) - The text to be anonymized.
    output: anonymized_text (str) - The anonymized text.
    '''
    try:
        entities = flair_NER_German(text)
        if entities:
            anonymized_text = text
            for entity in entities:
                if entity.tag == 'PER':
                    person = get_random_name()
                    anonymized_text = anonymized_text.replace(entity.text, 'PERSON')
            return anonymized_text
        else:
            return text
    except Exception as e:
        logger.error(f"Anonymizing text failed: {e}")
        return None
    
