from __future__ import annotations

from pathlib import Path
from typing import overload

import cv2
import numpy as np
from PIL import Image


@overload
def load_image_into_np(image_input: str | Path) -> np.ndarray: ...
@overload
def load_image_into_np(image_input: Image.Image) -> np.ndarray: ...
@overload
def load_image_into_np(image_input: np.ndarray) -> np.ndarray: ...


def load_image_into_np(
    image_input: str | Path | Image.Image | np.ndarray,
) -> np.ndarray:
    if isinstance(image_input, np.ndarray):
        if image_input.ndim not in (2, 3):
            raise ValueError("NumPy image must have 2 or 3 dimensions")
        return image_input.copy()

    if isinstance(image_input, Image.Image):
        # PIL RGB -> OpenCV BGR
        arr = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    path = Path(image_input).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image
