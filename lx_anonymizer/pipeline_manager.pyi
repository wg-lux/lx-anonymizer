from pathlib import Path

type ModifiedImageMap = dict[tuple[str, str], str]

def process_images_with_OCR_and_NER(
    file_path: Path | str,
    east_path: str = ...,
    device: str = ...,
    min_confidence: float = ...,
    width: int = ...,
    height: int = ...,
    skip_blur: bool = ...,
    skip_reassembly: bool = ...,
) -> tuple[ModifiedImageMap, dict[str, object]]: ...
