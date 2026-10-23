from typing import Optional

from spacy.language import Language

from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.ner.spacy_extractor import SpacyModelManager


EntityTuple = tuple[str, int, int, str]

nlp: Optional[Language] = None

logger = get_logger(__name__)


def spacy_NER_German(text: object) -> Optional[list[EntityTuple]]:
    global nlp

    if not isinstance(text, str):
        logger.error("NER input must be a string")
        return None
    if nlp is None:
        logger.error("NER model is not loaded.")
        try:
            logger.info("Loading spaCy German NER model...")
            nlp = SpacyModelManager.get_model()
            logger.info("spaCy German NER model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load spaCy German NER model (%s)", type(e).__name__)
            nlp = None

    if nlp is None:
        return None

    try:
        doc = nlp(text)
        entities = [
            (ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents
        ]

        if entities:
            logger.info("NER extraction completed: entity_count=%d", len(entities))

            per_found = any(
                ent.label_ == "PER" or ent.label_ == "PERSON" for ent in doc.ents
            )

            if per_found:
                logger.info("A person tag ('PER') was found in the text. Replacing...")
            else:
                logger.info("No person tag ('PER') was found in the text.")
            return entities
        else:
            logger.info("No entities found")
            return None
    except Exception as e:
        logger.error("Error in NER_German (%s)", type(e).__name__)
        return None


def model_load() -> Optional[Language]:
    global nlp
    try:
        logger.info("Loading spaCy German NER model...")
        nlp = SpacyModelManager.get_model()
        logger.info("spaCy German NER model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load spaCy German NER model (%s)", type(e).__name__)
        nlp = None
    return nlp
