import fitz # fitz
import numpy as np
from pathlib import Path

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
    """Convert a PDF file to a list of image paths."""
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        # Use Path to create the image path
        image_path = Path(f"{pdf_path.stem}_page_{page_num}.png")
        pix.save(str(image_path))  # Save as string if required by pymupdf
        images.append(image_path)
    return images

def merge_pdfs(pdf_paths, output_path):
    """Merge multiple PDFs into a single PDF using PyMuPDF."""
    merged_doc = fitz.open()
    for path in pdf_paths:
        doc = fitz.open(path)
        merged_doc.insert_pdf(doc)
    merged_doc.save(str(output_path))  # Save as string if required by pymupdf

def convert_image_to_pdf(image_path, pdf_path):
    """Converts an image to a PDF using PyMuPDF."""
    img = fitz.Pixmap(str(image_path))  # Convert to string if needed
    doc = fitz.open()
    rect = img.rect
    page = doc.new_page(width=rect.width, height=rect.height)
    page.insert_image(rect, filename=str(image_path))  # Convert to string if needed
    doc.save(str(pdf_path))  # Save as string if required by pymupdf
