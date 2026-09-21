from typing import Literal, TypedDict, overload

from PIL.Image import Image

from lx_anonymizer.runtime_types import ImageArray

class TesseractData(TypedDict):
    text: list[str]
    conf: list[int | str]
    left: list[int | str]
    top: list[int | str]
    width: list[int | str]
    height: list[int | str]
    block_num: list[int | str]

class Output:
    DICT: Literal["dict"]
    STRING: Literal["string"]
    BYTES: Literal["bytes"]

@overload
def image_to_string(
    image: Image | ImageArray | str,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    output_type: Literal["string"] = "string",
    timeout: float = 0,
) -> str: ...
@overload
def image_to_string(
    image: Image | ImageArray | str,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    *,
    output_type: Literal["bytes"],
    timeout: float = 0,
) -> bytes: ...
@overload
def image_to_data(
    image: Image | ImageArray | str,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    *,
    output_type: Literal["dict"],
    timeout: float = 0,
) -> TesseractData: ...
@overload
def image_to_data(
    image: Image | ImageArray | str,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    output_type: Literal["string"] = "string",
    timeout: float = 0,
) -> str: ...
