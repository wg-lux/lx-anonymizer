from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pymupdf  # type: ignore[import-untyped]
from PIL import Image


class _Pixmap(Protocol):
    samples: bytes
    width: int
    height: int
    n: int
    alpha: bool


class _PdfPage(Protocol):
    def get_pixmap(self) -> _Pixmap: ...

    def insert_image(self, rect: object, filename: str) -> None: ...


class _PdfDocument(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _PdfPage: ...

    def insert_pdf(self, doc: _PdfDocument) -> None: ...

    def new_page(self, width: float, height: float) -> _PdfPage: ...

    def save(self, filename: str) -> None: ...


def convert_pdf_page_to_image(page: _PdfPage) -> np.ndarray:
    """Convert a single PDF page into a NumPy image array."""
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    return img


def convert_pdf_to_images(pdf_path: str | Path) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Image objects."""
    if not isinstance(pdf_path, Path):
        pdf_path = Path(pdf_path)

    doc = cast(_PdfDocument, pymupdf.open(str(pdf_path)))
    images: list[Image.Image] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        mode = "RGBA" if pix.alpha else "RGB"
        pil_image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        images.append(pil_image)

    return images


def merge_pdfs(pdf_paths: list[str | Path], output_path: str | Path) -> None:
    """Merge multiple PDFs into a single PDF using PyMuPDF."""
    merged_doc = cast(_PdfDocument, pymupdf.open())
    for path in pdf_paths:
        doc = cast(_PdfDocument, pymupdf.open(str(path)))
        merged_doc.insert_pdf(doc)
    merged_doc.save(str(output_path))


def convert_image_to_pdf(image_path: str | Path, pdf_path: str | Path) -> None:
    """Convert an image to a PDF using PyMuPDF."""
    doc = cast(_PdfDocument, pymupdf.open())
    with Image.open(image_path) as image:
        width, height = image.size
    page = doc.new_page(width=float(width), height=float(height))
    rect = pymupdf.Rect(0, 0, width, height)
    page.insert_image(rect, filename=str(image_path))
    doc.save(str(pdf_path))
