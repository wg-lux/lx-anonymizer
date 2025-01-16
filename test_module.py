import pytest

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
    from lx_anonymizer.spacy_NER import spacy_NER_German
    text = "Herr Müller war heute in Berlin."
    entities = spacy_NER_German(text)
    assert entities==[('Herr Müller', 0, 10, 'PER'), ('Berlin', 20, 26, 'LOC')]