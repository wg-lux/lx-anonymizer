"""Shared in-process contracts; serialized domain models belong to lx_dtypes.

Box uses pixel corner coordinates (x1, y1, x2, y2). OpenCV rectangles
(x, y, width, height) must be converted before crossing that boundary.
ImageArray covers grayscale and color uint8 buffers; PixelArray also supports
source DICOM pixel depths. Neither alias asserts a channel layout.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Protocol, TypeAlias, TypedDict, overload

import numpy as np
import numpy.typing as npt

Box: TypeAlias = tuple[int, int, int, int]
ImageArray: TypeAlias = npt.NDArray[np.uint8]
PixelArray: TypeAlias = npt.NDArray[np.generic]
OcrResult: TypeAlias = tuple[str, Box]
ModifiedImageMap: TypeAlias = dict[tuple[str, str], str]


@dataclass(frozen=True)
class VideoExecutionHooks:
    """Attempt-local host integration for canonical video frame provenance.

    The encoder runner must raise on failure. The decoder observer encloses
    decoding and finalization; each callback identifies a source frame written
    to the output. Hosts own evidence persistence, never the masking algorithm.
    """

    run_encoder: Callable[[list[str]], CompletedProcess[str]]
    observe_decoded_frames: Callable[
        [Path, Path], AbstractContextManager[Callable[[int], None]]
    ]


class OcrConfig(TypedDict):
    """Required Tesseract invocation options shared by every frame OCR backend."""

    lang: str
    psm: int
    oem: int
    dpi: int


class NamedOcrConfig(OcrConfig):
    """A diagnostic configuration with its required display name."""

    name: str


class DnnNet(Protocol):
    """OpenCV returns one tensor by default or one tensor per requested output."""

    def setInput(self, blob: PixelArray) -> None: ...

    @overload
    def forward(self) -> PixelArray: ...

    @overload
    def forward(self, outNames: list[str]) -> tuple[PixelArray, ...]: ...


class Cv2DnnModule(Protocol):
    def readNet(self, model_path: str) -> DnnNet: ...

    def blobFromImage(
        self,
        image: PixelArray,
        scalefactor: float,
        size: tuple[int, int],
        mean: tuple[float, float, float],
        swapRB: bool,
        crop: bool,
    ) -> PixelArray: ...

    def NMSBoxes(
        self,
        bboxes: list[list[int]],
        scores: list[float],
        score_threshold: float,
        nms_threshold: float,
    ) -> PixelArray: ...


class PydicomReader(Protocol):
    """Optional DICOM reader; consumers validate the returned dataset at the edge."""

    def dcmread(self, path: str | Path, **kwargs: object) -> object: ...
