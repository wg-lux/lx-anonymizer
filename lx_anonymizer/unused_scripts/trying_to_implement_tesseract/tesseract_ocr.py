import importlib

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
# import fitz
try:
    import pypdf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pypdf = None  # type: ignore[assignment]
import os
import cv2
import numpy as np
import spacy
import re
from spacy.tokens import Doc
import json


def download_model(model_name):
    download_fn = getattr(spacy.util, "download_model", None)
    if download_fn is not None:
        download_fn(model_name)
    else:
        raise ImportError("spaCy download utility not available")

os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
tessdata_dir_config = '--tessdata-dir "/opt/homebrew/share/tessdata"'


# load a spaCy model
try:
    nlp = spacy.load("de_core_news_lg")
except OSError:
    print("Model not found. Downloading...")
    download_model("de_core_news_lg")
    nlp = spacy.load("de_core_news_lg")


def run_script():
    return "Processing completed"

def replace_unwanted_characters(word):
    # Erlaube nur Zeichen a-z, A-Z, ä, ü, ö, Ä, Ü, Ö, ß, 1-9, . , ' / & % ! " < > + * # ( ) € und -
    allowed_chars = r"[^a-zA-ZäüöÄÜÖß0-9.,'`/&%!\"<>+*#()\€_:-]"
    return re.sub(allowed_chars, '', word)


def clean_newlines(text):
    # Replace two or more consecutive "\n" characters with a single "\n"
    cleaned_text = re.sub(r'\n{2,}', '\n', text)

    # Replace remaining "\n" characters with a space
    cleaned_text = cleaned_text.replace("\n", " ")

    return cleaned_text

def process_image(image, use_mock=False):
    #print("Image size:", image.size)
    #print("Image format:", image.format)
    
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image passed to process_image")

    if use_mock:
        # Return a simple mock image     
        image = Image.new('RGB', (100, 100), color='red')
        return image
    
    # Convert image to OpenCV format
    image = np.array(image)

    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarization
    #_,  image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Scale image
    desired_width = 5000
    aspect_ratio = image.shape[1] / image.shape[0] # width / height
    desired_height = int(desired_width / aspect_ratio)

    image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_AREA)    
    
    # Increase contrast using histogram equalization
    #image = cv2.equalizeHist(image)

    # Noise reduction with a Median filter
    #image = cv2.medianBlur(image, 1)

    # Skew correction
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Convert back to PIL.Image format
    image = Image.fromarray(image)
      # Increase contrast
    #enhancer = ImageEnhance.Contrast(image)
    #image = enhancer.enhance(0.9)
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(0.2) 
    
    # Apply blurring filter
    #image = image.filter(ImageFilter.GaussianBlur(radius=0.03))
    
    return image

def convert_pdf_to_images(pdf_data):
    if pypdf is None:
        raise ImportError("pypdf is required to convert PDFs to images")

    images = []

    with open(str(pdf_data), 'rb') as file:
        reader = pypdf.PdfReader(file)
        num_pages = len(reader.pages)

        for i in range(num_pages):
            page = reader.pages[i]
            img = page.to_image(resolution=300)
            images.append(img)

    return images

def scale_coordinates(coords, image_size):
    # Convert the fractional coordinates into actual pixel values
    left = coords['left'] * image_size[0]
    top = coords['top'] * image_size[1]
    right = coords['right'] * image_size[0]
    bottom = coords['bottom'] * image_size[1]
    #print("scaled coordinates:", left, top, right, bottom)
    
    return (left, top, right, bottom)

def crop_image(image, coordinates):
    coordinates = scale_coordinates(coordinates, image.size)
    return image.crop(coordinates)

def extract_coordinates(htmlwithcoords):
    try:
        bs4_module = importlib.import_module("bs4")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("BeautifulSoup (bs4) is required for HTML parsing in this script") from exc

    BeautifulSoup = getattr(bs4_module, "BeautifulSoup")

    soup = BeautifulSoup(htmlwithcoords, 'html.parser')
    coordinates = []
    for word in soup.find_all(class_='ocrx_word'):
        bbox = word['title'].split(';')[0].split(' ')[1:]
        left, top, right, bottom = map(int, bbox)
        coordinates.append({'left': left, 'top': top, 'right': right, 'bottom': bottom})
    return coordinates

def process_text(extracted_text):
    extracted_text = clean_newlines(extracted_text)
    return extracted_text

    

def process_files(file_path):
    results = []

    print("Processing file:", file_path)


    with open(str(file_path), 'rb') as file:
        filename = file_path
        file_extension = filename.split('.')[-1].lower()

        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'tiff': 'image/tiff',
            'pdf': 'application/pdf',
        }

        file_type = mime_types.get(file_extension, 'application/octet-stream').split('/')[-1]
        if file_type not in ['jpg', 'jpeg', 'png', 'pdf', 'tiff']:
            raise ValueError('Invalid file type.')

        if file_type == 'pdf':
            pdf_data = file.read()
            images = convert_pdf_to_images(pdf_data)
            extracted_text = ''

            for img in images:
                img = process_image(img)
                if img is None or img.size == (0, 0):
                    raise ValueError("Failed to convert PDF page to image")
                text = pytesseract.image_to_string(img, lang='deu')
                extracted_text += text
            extracted_text = process_text(extracted_text)
        else:
            image = Image.open(file_path)
            image = process_image(image)

            try:
                extracted_text = pytesseract.image_to_string(image)
            except pytesseract.pytesseract.TesseractError as e:
                print(f"An error occurred during OCR processing: {e}")
                extracted_text = ""  

            extracted_text = process_text(extracted_text)

        result = {
            'filename': filename,
            'file_type': file_type,
            'extracted_text': extracted_text,
        }
        results.append(result)
        print("text detected:", result)

    return results


# Example usage:
#files = ['image1.jpg', 'document1.pdf']  # Replace with actual paths to your files
#coordinates_list = [[100, 100, 400, 400], [50, 50, 200, 200]]  # Example coordinates

# Process the files
##processed_results = process_files(files, coordinates_list)

# Output the results
#for result in processed_results:
#    print(json.dumps(result, ensure_ascii=False, indent=2))
