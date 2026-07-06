from pathlib import Path

def add_name_to_image(
    first_name: str,
    last_name: str,
    gender_par: str,
    first_name_box: tuple[int, int, int, int],
    last_name_box: tuple[int, int, int, int],
    device: str | None = ...,
    font: str | None = ...,
    font_size: int = ...,
    background_color: str = ...,
    font_color: str = ...,
    text_formatting: str = ...,
    line_spacing: int = ...,
    font_scale: int = ...,
    font_thickness: int = ...,
) -> Path: ...
def add_full_name_to_image(
    name: str,
    gender_par: str,
    box: tuple[int, int, int, int],
    font: str | None = ...,
    font_size: int = ...,
    background_color: tuple[int, int, int] = ...,
    font_color: tuple[int, int, int] = ...,
    font_scale: int = ...,
    font_thickness: int = ...,
) -> Path: ...
def add_device_name_to_image(
    name: str,
    gender_par: str,
    device: str | None = ...,
    font: str | None = ...,
    font_size: int = ...,
    background_color: tuple[int, int, int] = ...,
    font_color: tuple[int, int, int] = ...,
    line_spacing: int = ...,
    font_scale: int = ...,
    font_thickness: int = ...,
) -> Path: ...
