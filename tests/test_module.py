import pytest
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lx_anonymizer')))

os.environ['PYTHONPATH'] = '../lx_anonymizer'

def test_environment():
    try:
        import sys
        import cv2
        import numpy as np
        import torch
        import torchvision
        import pytesseract
        import spacy
        import flair
        imports = True
    except:
        imports = False
    assert imports==True

def test_ner():
    text = "Hans Müller war heute in Berlin."
    from lx_anonymizer.spacy_NER import spacy_NER_German
    entities = spacy_NER_German(text)
    # Check if 'Hans Müller' is found as an entity
    assert any(ent[0] == "Hans Müller" and ent[3] == "PER" for ent in entities)

def test_flair_ner():
    from lx_anonymizer.flair_NER import flair_NER_German
    text = "Hans Müller war heute in Berlin."
    entities = flair_NER_German(text)
    # Check if 'Hans Müller' is found as an entity
    assert any(
        span.text == "Hans Müller" and span.tag == "PER"
        for span in entities
    )
