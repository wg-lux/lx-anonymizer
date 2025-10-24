import cv2
import numpy as np
import uuid
from pathlib import Path
import time
import ast
from .device_reader import read_device, read_text_formatting
from .directory_setup import create_temp_directory
from .custom_logger import get_logger

logger = get_logger(__name__)

# Create or read temporary directory
temp_dir, base_dir, csv_dir = create_temp_directory()


def upscale_image(image, scale_factor=2):
    """
    Upscale an image by the given factor using INTER_CUBIC for sharper edges.
    """
    height, width = image.shape[:2]
    new_width = width * scale_factor
    new_height = height * scale_factor
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def compute_uniform_font_scale(text1, text2, font, box1, box2, font_thickness):
    """
    Compute a single font scale so that both text1 and text2 can fit
    into their respective boxes at the same scale.
    """
    # Extract dimensions from the boxes
    (startX1, startY1, endX1, endY1) = box1
    (startX2, startY2, endX2, endY2) = box2

    width1 = endX1 - startX1
    height1 = endY1 - startY1
    width2 = endX2 - startX2
    height2 = endY2 - startY2

    # We will find a scale that allows BOTH text1 and text2 to fit.
    font_scale = 1.0
    while font_scale > 0.1:
        size1 = cv2.getTextSize(text1, font, font_scale, font_thickness)[0]  # (width, height)
        size2 = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]

        # We check if both texts can fit into their boxes at this scale
        if size1[0] <= width1 and size1[1] <= height1 and size2[0] <= width2 and size2[1] <= height2:
            break
        font_scale -= 0.1

    return font_scale


def format_name(name, format_string):
    names = name.split()
    if len(names) < 2:
        return name
    first_name = names[0]
    last_name = " ".join(names[1:])
    formatted_name = format_string.replace("first_name", first_name).replace("last_name", last_name).replace("\n", "\n")
    return formatted_name


def validate_coordinates(coords):
    if not isinstance(coords, tuple) or len(coords) != 4:
        logger.error(f"Invalid coordinates format. Expected a tuple of four elements.")
        raise ValueError("Invalid coordinates format. Expected a tuple of four elements.")
    if not all(isinstance(coord, int) and coord >= 0 for coord in coords):
        logger.error(f"Coordinates must be non-negative integers.")
        raise ValueError("Coordinates must be non-negative integers.")


def draw_text_with_line_break(text, font, font_scale, font_color, font_thickness, background_color, first_name_coords, last_name_coords, line_spacing=20):
    validate_coordinates(first_name_coords)
    validate_coordinates(last_name_coords)

    padding = 10  # Padding around the text
    total_height = first_name_coords[3] + last_name_coords[3] + line_spacing + padding
    max_text_width = max(first_name_coords[2], last_name_coords[2])
    total_width = max_text_width + 2 * padding

    text_img = np.full((total_height, total_width, 3), background_color, dtype=np.uint8)

    names = text.split("\n")
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ""

    first_name_x = padding
    first_name_y = padding + first_name_coords[3]  # Adjusted for padding
    last_name_x = padding
    last_name_y = first_name_y + last_name_coords[3] + line_spacing  # Adjusted for line spacing

    cv2.putText(text_img, first_name, (first_name_x, first_name_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
    if last_name:
        cv2.putText(text_img, last_name, (last_name_x, last_name_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    return text_img


def draw_text_without_line_break(text, font, font_scale, font_color, font_thickness, background_color, first_name_coords, last_name_coords, line_spacing):
    validate_coordinates(first_name_coords)
    validate_coordinates(last_name_coords)

    padding = 10  # Padding around the text
    total_height = max(first_name_coords[3], last_name_coords[3]) + padding
    max_text_width = max(first_name_coords[2], last_name_coords[2])
    total_width = max_text_width + padding

    text_img = np.full((total_height, total_width, 3), background_color, dtype=np.uint8)

    names = text.split("\n")
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ""

    first_name_x = padding // 2
    first_name_y = padding // 2 + first_name_coords[3]  # Adjusted for padding
    last_name_x = padding // 2
    last_name_y = first_name_y + last_name_coords[3] + line_spacing  # Adjusted for line spacing

    cv2.putText(text_img, first_name, (first_name_x, first_name_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
    if last_name:
        cv2.putText(text_img, last_name, (last_name_x, last_name_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    return text_img


def draw_free_text(text, font, font_scale, font_color, font_thickness, background_color, first_name_coords, last_name_coords, line_spacing):
    validate_coordinates(first_name_coords)
    validate_coordinates(last_name_coords)

    padding = 10  # Padding around the text
    total_height = max(first_name_coords[3], last_name_coords[3]) + padding
    max_text_width = max(first_name_coords[2], last_name_coords[2])
    total_width = max_text_width + padding

    text_img = np.full((total_height, total_width, 3), background_color, dtype=np.uint8)

    names = text.split("\n")
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ""

    first_name_x = padding // 2
    first_name_y = first_name_coords[3]  # Reduced top padding
    last_name_x = padding // 2
    last_name_y = last_name_coords[3] + line_spacing  # Adjusted for line spacing

    cv2.putText(text_img, first_name, (first_name_x, first_name_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
    if last_name:
        cv2.putText(text_img, last_name, (last_name_x, last_name_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    return text_img


def add_device_name_to_image(
    name,
    gender_par,
    device=None,
    font=None,
    font_size=100,
    background_color=(0, 0, 0),
    font_color=(255, 255, 255),
    text_formatting=None,
    line_spacing=40,
    font_scale=1,
    font_thickness=1,  # Set thickness = 1 for better readability
):
    try:
        device_config = read_device(device)
        if device_config is None:
            logger.error(f"No configuration found for device: {device}")
            raise ValueError(f"No configuration found for device: {device}")

        (
            background_color,
            font_color,
            font,
            font_scale,
            font_thickness,
            text_formatting,
            first_name_x,
            first_name_y,
            first_name_width,
            first_name_height,
            last_name_x,
            last_name_y,
            last_name_width,
            last_name_height,
        ) = device_config

        # Convert from string to tuple if necessary
        background_color = ast.literal_eval(background_color) if isinstance(background_color, str) else background_color
        font_color = ast.literal_eval(font_color) if isinstance(font_color, str) else font_color

        # Force thickness = 1 for better readability
        font_thickness = 1

        first_name_coords = (first_name_x, first_name_y, first_name_width, first_name_height)
        last_name_coords = (last_name_x, last_name_y, last_name_width, last_name_height)
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"Error reading device configuration: {e}. Using default parameters.")
        background_color = (0, 0, 0)
        font_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        text_formatting = "first_name last_name"
        first_name_coords = (50, 50, 200, 50)
        last_name_coords = (50, 110, 200, 50)

    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX

    formatted_name = format_name(name, text_formatting)
    if "\n" in formatted_name:
        text_img = draw_text_with_line_break(
            formatted_name, font, font_scale, font_color, font_thickness, background_color, first_name_coords, last_name_coords, line_spacing
        )
    else:
        text_img = draw_text_without_line_break(
            formatted_name, font, font_scale, font_color, font_thickness, background_color, first_name_coords, last_name_coords, line_spacing
        )

    # Optional Upscale to reduce aliasing (then optionally downscale)
    # text_img = upscale_image(text_img, scale_factor=2)

    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"{gender_par}_{int(time.time())}_{unique_id}.png"
    output_image_path = Path(temp_dir) / output_filename
    cv2.imwrite(str(output_image_path), text_img)
    logger.debug(f"Temporary name image from device config saved to {output_image_path}")

    return output_image_path


def draw_text_to_fit(text, font, box, font_color, font_thickness, background_color):
    (startX, startY, endX, endY) = box
    box_width = endX - startX
    box_height = endY - startY

    # Reduced thickness = 1 for readability
    font_thickness = 1

    # Add padding
    padding = 10
    effective_width = box_width - (2 * padding)
    effective_height = box_height - (2 * padding)

    # Create slightly larger image to prevent cutoff
    text_img = np.full((box_height, box_width + 2 * padding, 3), background_color, dtype=np.uint8)

    # Find optimal font scale (start from 1.0 downwards)
    font_scale = 1.0
    while font_scale > 0.1:
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        if text_size[0] <= effective_width and text_size[1] <= effective_height:
            break
        font_scale -= 0.1

    # Calculate centered position
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = padding + (effective_width - text_size[0]) // 2
    text_y = padding + (effective_height + text_size[1]) // 2

    cv2.putText(text_img, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    # Trim any excess padding while keeping text centered
    non_zero_cols = np.where(text_img.any(axis=0).any(axis=1))[0]
    if len(non_zero_cols) > 0:
        left_margin = non_zero_cols[0]
        right_margin = text_img.shape[1] - non_zero_cols[-1]
        trim_amount = min(left_margin, right_margin)
        if trim_amount > 0:
            text_img = text_img[:, trim_amount:-trim_amount]

    logger.debug(f"Text drawn with size: {text_size}, position: ({text_x}, {text_y}), scale: {font_scale}")
    return text_img, font_scale


# Define letter size table (unchanged)
LETTER_SIZE_TABLE = {
    "a": (10, 20),
    "b": (10, 20),
    "c": (10, 20),
    "d": (10, 20),
    "e": (10, 20),
    "f": (5, 20),
    "g": (10, 20),
    "h": (10, 20),
    "i": (5, 20),
    "j": (5, 20),
    "k": (10, 20),
    "l": (5, 20),
    "m": (15, 20),
    "n": (10, 20),
    "o": (10, 20),
    "p": (10, 20),
    "q": (10, 20),
    "r": (10, 20),
    "s": (10, 20),
    "t": (5, 20),
    "u": (10, 20),
    "v": (10, 20),
    "w": (15, 20),
    "x": (10, 20),
    "y": (10, 20),
    "z": (10, 20),
    "A": (10, 20),
    "B": (10, 20),
    "C": (10, 20),
    "D": (10, 20),
    "E": (10, 20),
    "F": (10, 20),
    "G": (10, 20),
    "H": (10, 20),
    "I": (5, 20),
    "J": (5, 20),
    "K": (10, 20),
    "L": (10, 20),
    "M": (15, 20),
    "N": (10, 20),
    "O": (10, 20),
    "P": (10, 20),
    "Q": (10, 20),
    "R": (10, 20),
    "S": (10, 20),
    "T": (10, 20),
    "U": (10, 20),
    "V": (10, 20),
    "W": (15, 20),
    "X": (10, 20),
    "Y": (10, 20),
    "Z": (10, 20),
    " ": (5, 20),
    "0": (10, 20),
    "1": (5, 20),
    "2": (10, 20),
    "3": (10, 20),
    "4": (10, 20),
    "5": (10, 20),
    "6": (10, 20),
    "7": (10, 20),
    "8": (10, 20),
    "9": (10, 20),
}


def calculate_text_size(text, font_scale, font_thickness):
    width = 0
    height = 0
    for char in text:
        if char in LETTER_SIZE_TABLE:
            char_width, char_height = LETTER_SIZE_TABLE[char]
            width += int(char_width * font_scale)
            height = max(height, int(char_height * font_scale))
        else:
            size = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            width += size[0]
            height = max(height, size[1])
    logger.debug(f"Text size calculated by calculate_text_size: ({width}, {height})")
    return width, height


def enlarge_box(box, required_width, required_height, padding=10):
    startX, startY, endX, endY = box
    current_width = endX - startX
    current_height = endY - startY

    new_width = max(current_width, required_width + 2 * padding)
    new_height = max(current_height, required_height + 2 * padding)

    new_startX = startX - (new_width - current_width) // 2
    new_startY = startY - (new_height - current_height) // 2
    new_endX = new_startX + new_width
    new_endY = new_startY + new_height

    logger.debug(f"Changed box coordinates from ({startX}, {startY}, {endX}, {endY}) to ({new_startX}, {new_startY}, {new_endX}, {new_endY})")

    return new_startX, new_startY, new_endX, new_endY


def draw_text_centered(text, font, font_scale, font_color, font_thickness, background_color, box, padding=10):
    # Calculate required text size
    text_width, text_height = calculate_text_size(text, font_scale, font_thickness)

    # Enlarge box if necessary
    startX, startY, endX, endY = enlarge_box(box, text_width, text_height, padding)
    box_width = endX - startX
    box_height = endY - startY

    text_img = np.full((box_height, box_width, 3), background_color, dtype=np.uint8)

    # Calculate center alignment with padding
    text_x = (box_width - text_width) // 2
    text_y = (box_height + text_height) // 2

    # Draw text
    cv2.putText(text_img, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
    logger.debug(f"Text drawn centrally: Text size: {text_width}, {text_height}, position: ({text_x}, {text_y}), scale: {font_scale}")

    return text_img


def add_name_to_image(
    first_name,
    last_name,
    gender_par,
    first_name_box,
    last_name_box,
    device=None,
    font=None,
    font_size=100,
    background_color="(255, 255, 255)",
    font_color="(0, 0, 0)",
    text_formatting="first_name last_name",
    line_spacing=40,
    font_scale=1,
    font_thickness=1,  # reduce thickness
):
    logger.info(f"Adding name to image: {first_name} {last_name}")

    try:
        config = read_text_formatting(device)
        if config is None:
            raise ValueError(f"No text formatting configuration found for device: {device}")

        background_color, font_color, font, font_scale, font_thickness, text_formatting = config
        background_color = ast.literal_eval(background_color) if isinstance(background_color, str) else background_color
        font_color = ast.literal_eval(font_color) if isinstance(font_color, str) else font_color
        # Force thickness = 1 for uniform readability
        font_thickness = 1
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.debug(f"Error reading device configuration: {e}. Using default parameters.")
        background_color = (255, 255, 255)
        font_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        text_formatting = "first_name last_name"

    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX

    # Make sure both first_name and last_name use the SAME scale
    # by computing one scale that fits both boxes
    uniform_scale = compute_uniform_font_scale(first_name, last_name, font, first_name_box, last_name_box, font_thickness)

    # Standardize box heights
    standard_height = max(first_name_box[3] - first_name_box[1], last_name_box[3] - last_name_box[1])

    fixed_spacing = 5  # pixels

    first_name_standardized = (first_name_box[0], first_name_box[1], first_name_box[2], first_name_box[1] + standard_height)
    last_name_standardized = (last_name_box[0], last_name_box[1], last_name_box[2], last_name_box[1] + standard_height)

    # Draw text with standardized heights and uniform scale
    text_img_fn, _ = draw_text_to_fit(first_name, font, first_name_standardized, font_color, font_thickness, background_color)
    text_img_ln, _ = draw_text_to_fit(last_name, font, last_name_standardized, font_color, font_thickness, background_color)

    # If you want to override with the uniform_scale found above, you can do so directly:
    # text_img_fn = draw_text_centered(first_name, font, uniform_scale, font_color, font_thickness, background_color, first_name_standardized)
    # text_img_ln = draw_text_centered(last_name, font, uniform_scale, font_color, font_thickness, background_color, last_name_standardized)

    total_width = text_img_fn.shape[1] + fixed_spacing + text_img_ln.shape[1]
    final_img = np.full((standard_height, total_width, 3), background_color, dtype=np.uint8)

    # Place first name
    final_img[:, : text_img_fn.shape[1]] = cv2.resize(text_img_fn, (text_img_fn.shape[1], standard_height))

    # Place last name after spacing
    start_x = text_img_fn.shape[1] + fixed_spacing
    final_img[:, start_x : start_x + text_img_ln.shape[1]] = cv2.resize(text_img_ln, (text_img_ln.shape[1], standard_height))

    # Optional Upscale for less aliasing (then you could downscale back if you prefer):
    # final_img = upscale_image(final_img, scale_factor=2)

    unique_id = str(uuid.uuid4())
    output_filename = f"{gender_par}_{int(time.time())}_{unique_id}.png"
    output_image_path = Path(temp_dir) / output_filename
    cv2.imwrite(str(output_image_path), final_img)

    logger.debug(f"Name added to image. Image saved to {output_image_path}")
    logger.info(f"Image saved to {output_image_path}")

    return output_image_path


def hconcat_resize_min_with_spacing(im_list, spacing=10, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for im in im_list]

    total_width = sum(im.shape[1] for im in im_list_resize) + (len(im_list_resize) - 1) * spacing
    result_image = np.full((h_min, total_width, 3), 255, dtype=np.uint8)

    current_x = 0
    for image in im_list_resize:
        result_image[:, current_x : current_x + image.shape[1]] = image
        current_x += image.shape[1] + spacing

    return result_image


def add_full_name_to_image(
    name, gender_par, box, font=None, font_size=100, background_color=(0, 0, 0), font_color=(255, 255, 255), font_scale=1, font_thickness=1
):
    StartX, StartY, EndX, EndY = box
    box_width = EndX - StartX

    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX

    # Use the updated draw_text_to_fit with lineType=LINE_AA
    text_img, actual_scale = draw_text_to_fit(name, font, box, font_color, font_thickness, background_color)

    # If the text overflows, create a larger canvas
    text_size = cv2.getTextSize(name, font, actual_scale, font_thickness)[0]
    if text_size[0] > box_width:
        new_width = text_size[0] + 20  # some padding
        larger_text_img = np.full((text_img.shape[0], new_width, 3), background_color, dtype=np.uint8)
        larger_text_img[:, : text_img.shape[1]] = text_img
        text_img = larger_text_img

    # Optional upscale
    # text_img = upscale_image(text_img, scale_factor=2)

    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"{gender_par}_{int(time.time())}_{unique_id}.png"
    output_image_path = Path(temp_dir) / output_filename
    cv2.imwrite(str(output_image_path), text_img)
    logger.info(f"Image saved to {output_image_path}")

    return output_image_path
