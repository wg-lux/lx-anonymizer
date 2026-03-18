from importlib import import_module
from typing import Any

from lx_anonymizer.setup.custom_logger import get_logger

logger = get_logger(__name__)
"""
Flair NER Tagger
This script loads a Flair Named Entity Recognition (NER) tagger for German text.
The Text is searched for entities and the entities are returned, when a PER, 
or Person Tag is found.
input: text (str) - The text to be analyzed for entities.
output: entities (List) - A list of entities found in the text.
"""

tagger = None
_flair_import_error = None
SentenceType: Any = None
SequenceTaggerType: Any = None

try:
    SentenceType = getattr(import_module("flair.data"), "Sentence")
    SequenceTaggerType = getattr(import_module("flair.models"), "SequenceTagger")
except ImportError as exc:
    _flair_import_error = exc


def _get_tagger():
    global tagger
    if tagger is not None:
        return tagger
    if SequenceTaggerType is None:
        logger.info(
            "Flair is not installed. Install with: pip install lx-anonymizer[nlu]"
        )
        return None
    try:
        logger.info("Loading Flair German NER tagger...")
        tagger = SequenceTaggerType.load("flair/ner-german")
        logger.info("Flair German NER tagger loaded successfully.")
        return tagger
    except Exception as e:
        logger.error(f"Failed to load Flair German NER tagger: {e}")
        return None


def flair_NER_German(text):
    if not isinstance(text, str):
        logger.error(f"Expected a string, but got {type(text)}")
        return None
    active_tagger = _get_tagger()
    if active_tagger is None or SentenceType is None:
        logger.error("NER tagger is not loaded.")
        return None
    try:
        sentence = SentenceType(text)
        active_tagger.predict(sentence)
        entities = sentence.get_spans("ner")
        if entities:
            logger.info("The following NER tags are found:")
            for entity in entities:
                logger.info(entity)

            per_found = any(entity.tag == "PER" for entity in entities)
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
        return None
