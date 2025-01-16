import pytest

def test_environment():
    try:
        import os
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
