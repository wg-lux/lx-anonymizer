import spacy
from custom_logger import get_logger

logger = get_logger(__name__)

try:
    logger.info("Loading spaCy German NER model...")
    nlp = spacy.load("de_core_news_md")
    logger.info("spaCy German NER model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy German NER model: {e}")
    nlp = None

def NER_German(text):
    if not isinstance(text, str):
        logger.error(f"Expected a string, but got {type(text)}")
        return None

    if nlp is None:
        logger.error("NER model is not loaded.")
        return None

    try:
        doc = nlp(text)
        entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        for token in doc:
            print(token.text, token.pos_, token.dep_)

        if entities:
            logger.info('The following NER tags are found:')
            for entity in entities:
                logger.info(entity)

            per_found = any(ent.label_ == 'PER' or ent.label_ =='PERSON' for ent in doc.ents)

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
