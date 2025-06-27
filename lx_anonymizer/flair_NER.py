from flair.data import Sentence
from flair.models import SequenceTagger
from .custom_logger import get_logger
logger = get_logger(__name__)
# Load the NER tagger once at module level
'''
Flair NER Tagger
This script loads a Flair Named Entity Recognition (NER) tagger for German text.
The Text is searched for entities and the entities are returned, when a PER, 
or Person Tag is found.
input: text (str) - The text to be analyzed for entities.
output: entities (List) - A list of entities found in the text.
'''
try:
    logger.info("Loading Flair German NER tagger...")
    tagger = SequenceTagger.load("flair/ner-german")  # Use the correct model name
    logger.info("Flair German NER tagger loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Flair German NER tagger: {e}")
    tagger = None
def flair_NER_German(text):
    if not isinstance(text, str):
        logger.error(f"Expected a string, but got {type(text)}")
        return None
    if tagger is None:
        logger.error("NER tagger is not loaded.")
        return None
    try:
        sentence = Sentence(text)
        tagger.predict(sentence)
        entities = sentence.get_spans('ner')
        if entities:
            logger.info('The following NER tags are found:')
            for entity in entities:
                logger.info(entity)
            
            per_found = any(entity.tag == 'PER' for entity in entities)
            if per_found:
                logger.info("A person tag ('PER') was found in the text. Replacing...")
            else:
                logger.info("No person tag ('PER') was found in the text.")
            return entities
        else:
            logger.info("No entities found")
            return None
    except Exception as e:
        logger.error(f"Error in NER_German: {e}")
        return None