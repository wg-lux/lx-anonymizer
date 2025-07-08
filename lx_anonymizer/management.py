from pathlib import Path
from .pipeline_manager import process_images_with_OCR_and_NER
from .text_anonymizer import anonymize_text

class PDF_Anonymizer:
    def __init__(self, pdf_path: Path, output_dir: Path):
        self.pdf_path = pdf_path
        self.output_dir = output_dir

    def anonymize(self):# -> Any | None:# -> Any | None:
        output = anonymize_text(self.pdf_path)
        return output
    
class Image_Anonymizer:
    def __init__(self, image_path: Path, output_dir: Path):
        self.image_path = image_path
        self.output_dir = output_dir

    def anonymize(self):# -> Any | None:# -> Any | None:
        output = process_images_with_OCR_and_NER(self.image_path)
        return output
    
        




