import pymupdf # pymupdf
import numpy as np
from pathlib import Path
from PIL import Image


'''
This script provides functions for performing various operations on PDF files and images using PyMuPDF.

Functions:
- convert_pdf_page_to_image(page): Convert a single PDF page into an image using PyMuPDF and encode it using OpenCV.
- convert_pdf_to_images(pdf_path): Convert a PDF file to a list of image paths.
- merge_pdfs(pdf_paths, output_path): Merge multiple PDFs into a single PDF using PyMuPDF.
- convert_image_to_pdf(image_path, pdf_path): Convert an image to a PDF using PyMuPDF.
'''

def convert_pdf_page_to_image(page):
    """
    Convert a single PDF page into an image using PyMuPDF and then encode it using OpenCV.
    """
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img

def convert_pdf_to_images(pdf_path):
    """
    Convert a PDF file to a list of PIL Image objects.
    """
    # Ensure pdf_path is a Path object
    if not isinstance(pdf_path, Path):
        pdf_path = Path(pdf_path)
    
    # Open the PDF using PyMuPDF (fitz)
    doc = pymupdf.open(str(pdf_path))
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to a pixmap
        pix = page.get_pixmap()
        
        # Determine the mode (use RGBA if the pixmap has an alpha channel)
        mode = "RGBA" if pix.alpha else "RGB"
        
        # Convert the pixmap samples to a PIL Image using frombytes
        pil_image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        images.append(pil_image)
    
    return images

def merge_pdfs(pdf_paths, output_path):
    """Merge multiple PDFs into a single PDF using PyMuPDF."""
    merged_doc = pymupdf.open()
    for path in pdf_paths:
        doc = pymupdf.open(path)
        merged_doc.insert_pdf(doc)
    merged_doc.save(str(output_path))  # Save as string if required by pymupdf

def convert_image_to_pdf(image_path, pdf_path):
    """Converts an image to a PDF using PyMuPDF."""
    img = pymupdf.Pixmap(str(image_path))  # Convert to string if needed
    doc = pymupdf.open()
    rect = img.rect
    page = doc.new_page(width=rect.width, height=rect.height)
    page.insert_image(rect, filename=str(image_path))  # Convert to string if needed
    doc.save(str(pdf_path))  # Save as string if required by pymupdf
