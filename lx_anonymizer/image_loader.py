from pathlib import Path
import fitz
from .custom_logger import get_logger

logger = get_logger(__name__)


def get_image_paths(image_or_pdf_path: Path, temp_dir: Path):
    image_paths = []

    if not temp_dir.exists() or not temp_dir.is_dir():
        raise ValueError(f"Temporary directory {temp_dir} does not exist or is not a directory.")

    if image_or_pdf_path.suffix.lower() == ".pdf":
        try:
            doc = fitz.open(str(image_or_pdf_path))  # fitz expects a string
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap()
                temp_img_path = Path(temp_dir) / f"page_{page_num}.png"
                pix.save(str(temp_img_path))  # fitz requires a string path
                image_paths.append(temp_img_path)
        except Exception as e:
            raise RuntimeError(f"Error processing PDF {image_or_pdf_path}: {e}")
    else:
        if not image_or_pdf_path.exists():
            raise FileNotFoundError(f"The file {image_or_pdf_path} does not exist.")
        image_paths.append(image_or_pdf_path)

    return image_paths
