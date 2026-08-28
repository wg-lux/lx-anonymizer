from pathlib import Path
from typing import Sequence

from lx_anonymizer.text_detection.phi_region_detector import PhiRegionDetector

type ModifiedImageMap = dict[tuple[str, str], str]
type BoundingBox = tuple[int, int, int, int]

def detect_combined_text_boxes(
    img_path: Path,
    east_path: str,
    min_confidence: float,
    width: int,
    height: int,
    region_detector: PhiRegionDetector | None = ...,
    phi_regions: Sequence[BoundingBox] | None = ...,
) -> list[BoundingBox]: ...
def process_images_with_OCR_and_NER(
    file_path: Path | str,
    east_path: str = ...,
    device: str = ...,
    min_confidence: float = ...,
    width: int = ...,
    height: int = ...,
    skip_blur: bool = ...,
    skip_reassembly: bool = ...,
    region_detector: PhiRegionDetector | None = ...,
) -> tuple[ModifiedImageMap, dict[str, object]]: ...
