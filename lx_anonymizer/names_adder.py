import cv2
import numpy as np
import uuid
from pathlib import Path
import time
import ast
from .device_reader import read_device, read_text_formatting
from .directory_setup import create_temp_directory
from .box_operations import make_box_from_device_list, make_box_from_name, extend_boxes_if_needed
from .custom_logger import get_logger

# --- NEW IMPORTS FOR PIL ---
from PIL import Image, ImageDraw, ImageFont

logger = get_logger(__name__)

# Create or read temporary directory
temp_dir, base_dir, csv_dir = create_temp_directory()

def numpy_to_pil(np_img_bgr):
    """
    Convert an OpenCV-style (BGR) NumPy array to a Pillow (RGB) Image.
    """
    return Image.fromarray(np_img_bgr[:, :, ::-1])  # flip BGR -> RGB

def pil_to_numpy(pil_img):
    """
    Convert a Pillow (RGB) Image to an OpenCV-style (BGR) NumPy array.
    """
    return np.array(pil_img)[:, :, ::-1]  # flip RGB -> BGR

def load_font(font_or_path, font_size=40):
    """
    Try loading a TTF font if given a path; 
    if 'font_or_path' is already an int (like cv2.FONT_HERSHEY_SIMPLEX),
    or if loading fails, fallback to a default Pillow font.
    """
    if isinstance(font_or_path, str):
        try:
            return ImageFont.truetype(font_or_path, font_size)
        except Exception:
            logger.warning(f"Could not load TTF font at '{font_or_path}'. Falling back to Pillow default.")
            return ImageFont.load_default()
    else:
        # If user passed in e.g. cv2.FONT_HERSHEY_SIMPLEX, just use default
        return ImageFont.load_default()

def upscale_image(image, scale_factor=2):
    """
    Upscale an image by the given factor using Pillow's resampling.
    """
    pil_img = Image.fromarray(image)  # This image is already RGB or BGR?
    width, height = pil_img.size
    new_width = width * scale_factor
    new_height = height * scale_factor
    pil_img = pil_img.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
    return np.array(pil_img)  # Return as NumPy array (RGB)

def compute_uniform_font_scale(text1, text2, font, box1, box2, font_thickness):
    """
    Dummy function for Pillow usage. In OpenCV you used `cv2.getTextSize` to gauge scale. 
    Here we won't do actual 'scaling' changes. We can approximate by stepping down in 0.1 increments 
    and checking if it fits using Pillow’s text metrics.
    """
    (startX1, startY1, endX1, endY1) = box1
    (startX2, startY2, endX2, endY2) = box2

    width1 = endX1 - startX1
    height1 = endY1 - startY1
    width2 = endX2 - startX2
    height2 = endY2 - startY2

    # Pillow 'font_thickness' doesn't directly matter. We'll approximate with a stepping approach.
    font_size = 100  # start high
    while font_size > 5:
        pil_font = load_font(font, font_size)
        # measure text1
        w1, h1 = measure_text_size_pil(text1, pil_font)
        # measure text2
        w2, h2 = measure_text_size_pil(text2, pil_font)
        if w1 <= width1 and h1 <= height1 and w2 <= width2 and h2 <= height2:
            # We'll interpret "font_scale" as the final "font_size" for usage
            return font_size
        font_size -= 5

    return 10  # some small fallback

def measure_text_size_pil(text, font):
    """
    Returns (width, height) of `text` using Pillow's textbbox.
    """
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top

def format_name(name, format_string):
    names = name.split()
    if len(names) < 2:
        return name
    first_name = names[0]
    last_name = ' '.join(names[1:])
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
    """
    Reimplementation using Pillow. Returns a NumPy array (RGB).
    """
    validate_coordinates(first_name_coords)
    validate_coordinates(last_name_coords)

    # create the bounding area
    padding = 10
    total_height = first_name_coords[3] + last_name_coords[3] + line_spacing + padding
    max_text_width = max(first_name_coords[2], last_name_coords[2])
    total_width = max_text_width + 2 * padding

    pil_img = Image.new("RGB", (total_width, total_height), color=background_color)
    draw = ImageDraw.Draw(pil_img)

    # We assume 'font_scale' is acting like a font size
    pil_font = load_font(font, int(font_scale if font_scale > 2 else font_scale * 30))
    names = text.split('\n')
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ''

    first_name_x = padding
    first_name_y = padding + first_name_coords[3]
    last_name_x = padding
    last_name_y = first_name_y + last_name_coords[3] + line_spacing

    draw.text((first_name_x, first_name_y), first_name, fill=font_color, font=pil_font)
    if last_name:
        draw.text((last_name_x, last_name_y), last_name, fill=font_color, font=pil_font)

    return np.array(pil_img)  # RGB

def draw_text_without_line_break(text, font, font_scale, font_color, font_thickness, background_color, first_name_coords, last_name_coords, line_spacing):
    """
    Pillow version that draws single-line text in separate regions. Returns a NumPy array (RGB).
    """
    validate_coordinates(first_name_coords)
    validate_coordinates(last_name_coords)

    padding = 10
    total_height = max(first_name_coords[3], last_name_coords[3]) + padding
    max_text_width = max(first_name_coords[2], last_name_coords[2])
    total_width = max_text_width + padding

    pil_img = Image.new("RGB", (total_width, total_height), color=background_color)
    draw = ImageDraw.Draw(pil_img)

    pil_font = load_font(font, int(font_scale if font_scale > 2 else font_scale * 30))

    names = text.split('\n')
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ''

    first_name_x = padding // 2
    first_name_y = padding // 2 + first_name_coords[3]
    last_name_x = padding // 2
    last_name_y = first_name_y + last_name_coords[3] + line_spacing

    draw.text((first_name_x, first_name_y), first_name, fill=font_color, font=pil_font)
    if last_name:
        draw.text((last_name_x, last_name_y), last_name, fill=font_color, font=pil_font)

    return np.array(pil_img)  # RGB

def draw_free_text(text, font, font_scale, font_color, font_thickness, background_color, first_name_coords, last_name_coords, line_spacing):
    """
    Another variant that returns a NumPy array. Uses Pillow for text.
    """
    validate_coordinates(first_name_coords)
    validate_coordinates(last_name_coords)

    padding = 10
    total_height = max(first_name_coords[3], last_name_coords[3]) + padding
    max_text_width = max(first_name_coords[2], last_name_coords[2])
    total_width = max_text_width + padding

    pil_img = Image.new("RGB", (total_width, total_height), color=background_color)
    draw = ImageDraw.Draw(pil_img)
    pil_font = load_font(font, int(font_scale if font_scale > 2 else font_scale * 30))

    names = text.split('\n')
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ''

    first_name_x = padding // 2
    first_name_y = first_name_coords[3]
    last_name_x = padding // 2
    last_name_y = last_name_coords[3] + line_spacing

    draw.text((first_name_x, first_name_y), first_name, fill=font_color, font=pil_font)
    if last_name:
        draw.text((last_name_x, last_name_y), last_name, fill=font_color, font=pil_font)

    return np.array(pil_img)  # RGB

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
    font_thickness=1
):
    try:
        device_config = read_device(device)
        if device_config is None:
            logger.error(f"No configuration found for device: {device}")
            raise ValueError(f"No configuration found for device: {device}")
        
        (background_color_str, font_color_str, font_path_str, font_scale_cfg, font_thickness_cfg,
         text_formatting_cfg, first_name_x, first_name_y, first_name_width, first_name_height, 
         last_name_x, last_name_y, last_name_width, last_name_height) = device_config

        background_color = ast.literal_eval(background_color_str) if isinstance(background_color_str, str) else background_color_str
        font_color = ast.literal_eval(font_color_str) if isinstance(font_color_str, str) else font_color_str
        font_scale = font_scale_cfg
        font_thickness = 1  # forced for readability
        text_formatting = text_formatting_cfg
        if font_path_str:
            font = font_path_str  # override

        first_name_coords = (first_name_x, first_name_y, first_name_width, first_name_height)
        last_name_coords = (last_name_x, last_name_y, last_name_width, last_name_height)
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"Error reading device configuration: {e}. Using default parameters.")
        background_color = (0, 0, 0)
        font_color = (255, 255, 255)
        font = None
        font_scale = 1
        font_thickness = 1
        text_formatting = "first_name last_name"
        first_name_coords = (50, 50, 200, 50)
        last_name_coords = (50, 110, 200, 50)

    if font is None:
        font = "arial.ttf"  # fallback

    formatted_name = format_name(name, text_formatting)
    if "\n" in formatted_name:
        text_img = draw_text_with_line_break(
            formatted_name, font, font_scale, font_color, font_thickness, background_color,
            first_name_coords, last_name_coords, line_spacing
        )
    else:
        text_img = draw_text_without_line_break(
            formatted_name, font, font_scale, font_color, font_thickness, 
            background_color, first_name_coords, last_name_coords, line_spacing
        )

    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"{gender_par}_{int(time.time())}_{unique_id}.png"
    output_image_path = Path(temp_dir) / output_filename

    # text_img is NumPy RGB array; we can save with OpenCV or Pillow
    cv2.imwrite(str(output_image_path), text_img[:, :, ::-1])  # flip RGB->BGR for cv2

    logger.debug(f"Temporary name image from device config saved to {output_image_path}")
    return output_image_path

def draw_text_to_fit(text, font, box, font_color, font_thickness, background_color):
    """
    Reimplementation with Pillow. Returns (np_array_rgb, font_scale_used).
    We'll guess a 'font_size' by stepping downward until it fits.
    """
    (startX, startY, endX, endY) = box
    box_width = endX - startX
    box_height = endY - startY

    # We'll do a stepping approach to find max font_size
    max_possible_size = 100
    best_font_size = 10
    for size in range(max_possible_size, 5, -2):
        test_font = load_font(font, size)
        w, h = measure_text_size_pil(text, test_font)
        if w <= box_width and h <= box_height:
            best_font_size = size
            break

    # Now draw with best_font_size
    pil_img = Image.new("RGB", (box_width, box_height), color=background_color)
    draw = ImageDraw.Draw(pil_img)
    final_font = load_font(font, best_font_size)
    w, h = measure_text_size_pil(text, final_font)

    # center
    x = (box_width - w) // 2
    y = (box_height - h) // 2
    draw.text((x, y), text, fill=font_color, font=final_font)

    # Return as NumPy array
    return np.array(pil_img), best_font_size

# Keep letter size table for consistency, even though we rely on Pillow
LETTER_SIZE_TABLE = {
    'a': (10, 20), 'b': (10, 20), 'c': (10, 20), 'd': (10, 20), 'e': (10, 20),
    'f': (5, 20), 'g': (10, 20), 'h': (10, 20), 'i': (5, 20), 'j': (5, 20),
    'k': (10, 20), 'l': (5, 20), 'm': (15, 20), 'n': (10, 20), 'o': (10, 20),
    'p': (10, 20), 'q': (10, 20), 'r': (10, 20), 's': (10, 20), 't': (5, 20),
    'u': (10, 20), 'v': (10, 20), 'w': (15, 20), 'x': (10, 20), 'y': (10, 20),
    'z': (10, 20), 'A': (10, 20), 'B': (10, 20), 'C': (10, 20), 'D': (10, 20),
    'E': (10, 20), 'F': (10, 20), 'G': (10, 20), 'H': (10, 20), 'I': (5, 20),
    'J': (5, 20), 'K': (10, 20), 'L': (10, 20), 'M': (15, 20), 'N': (10, 20),
    'O': (10, 20), 'P': (10, 20), 'Q': (10, 20), 'R': (10, 20), 'S': (10, 20),
    'T': (10, 20), 'U': (10, 20), 'V': (10, 20), 'W': (15, 20), 'X': (10, 20),
    'Y': (10, 20), 'Z': (10, 20), ' ': (5, 20), '0': (10, 20), '1': (5, 20),
    '2': (10, 20), '3': (10, 20), '4': (10, 20), '5': (10, 20), '6': (10, 20),
    '7': (10, 20), '8': (10, 20), '9': (10, 20)
}

def calculate_text_size(text, font_scale, font_thickness):
    """
    This is the old function name. We'll just measure using Pillow's approach with a 'default' font size.
    The letter table is no longer crucial if using Pillow directly.
    """
    # We'll pick an approximate base size
    base_font_size = int(font_scale if font_scale > 2 else font_scale*30)
    test_font = ImageFont.load_default()  # or load_font("arial.ttf", base_font_size)
    w, h = measure_text_size_pil(text, test_font)
    logger.debug(f"Text size calculated by PIL approach: ({w}, {h})")
    return w, h

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
    
    logger.debug(f"Changed box coordinates from ({startX}, {startY}, {endX}, {endY}) to "
                 f"({new_startX}, {new_startY}, {new_endX}, {new_endY})")
    return new_startX, new_startY, new_endX, new_endY

def draw_text_centered(text, font, font_scale, font_color, font_thickness, background_color, box, padding=10):
    """
    Pillow version of the center-draw. 
    We'll create a new image with size at least 'box' plus any needed expansions.
    Returns a NumPy array (RGB).
    """
    text_width, text_height = calculate_text_size(text, font_scale, font_thickness)
    startX, startY, endX, endY = enlarge_box(box, text_width, text_height, padding)
    box_width = endX - startX
    box_height = endY - startY

    pil_img = Image.new("RGB", (box_width, box_height), color=background_color)
    draw = ImageDraw.Draw(pil_img)

    # interpret font_scale as a font size
    final_font_size = int(font_scale if font_scale > 2 else font_scale*30)
    pil_font = load_font(font, final_font_size)

    x = (box_width - text_width) // 2
    y = (box_height - text_height) // 2

    draw.text((x, y), text, fill=font_color, font=pil_font)
    logger.debug(f"Text drawn centrally: ({text_width}x{text_height}), position=({x},{y}), scale={font_scale}")

    return np.array(pil_img)  # RGB

def hconcat_resize_min_with_spacing(im_list, spacing=10, interpolation=cv2.INTER_CUBIC):
    """
    This is specific to OpenCV usage. We'll keep it as is, 
    just ensure 'im' is a NumPy array in BGR or RGB. 
    """
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = []
    for im in im_list:
        # scale to h_min
        w = int(im.shape[1] * h_min / im.shape[0])
        resized = cv2.resize(im, (w, h_min), interpolation=interpolation)
        im_list_resize.append(resized)

    total_width = sum(img.shape[1] for img in im_list_resize) + (len(im_list_resize) - 1) * spacing
    result_image = np.full((h_min, total_width, 3), 255, dtype=np.uint8)

    current_x = 0
    for image in im_list_resize:
        result_image[:, current_x:current_x + image.shape[1]] = image
        current_x += image.shape[1] + spacing

    return result_image

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
    font_thickness=1
):
    logger.info(f"Adding name to image: {first_name} {last_name}")

    try:
        config = read_text_formatting(device)
        if config is None:
            raise ValueError(f"No text formatting configuration found for device: {device}")
        
        background_color, font_color, font_path, font_scale_cfg, font_thickness_cfg, text_formatting_cfg = config
        background_color = ast.literal_eval(background_color) if isinstance(background_color, str) else background_color
        font_color = ast.literal_eval(font_color) if isinstance(font_color, str) else font_color
        font = font_path or font  # override
        font_thickness = 1
        text_formatting = text_formatting_cfg
        font_scale = font_scale_cfg
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.debug(f"Error reading device configuration: {e}. Using default parameters.")
        background_color = (255, 255, 255)
        font_color = (0, 0, 0)
        font = "arial.ttf"
        font_scale = 1
        font_thickness = 1
        text_formatting = "first_name last_name"

    # unify scale
    uniform_scale = compute_uniform_font_scale(first_name, last_name, font, first_name_box, last_name_box, font_thickness)

    standard_height = max(first_name_box[3] - first_name_box[1],
                          last_name_box[3] - last_name_box[1])
    fixed_spacing = 5

    first_name_standardized = (
        first_name_box[0],
        first_name_box[1],
        first_name_box[2],
        first_name_box[1] + standard_height
    )
    last_name_standardized = (
        last_name_box[0],
        last_name_box[1],
        last_name_box[2],
        last_name_box[1] + standard_height
    )

    text_img_fn, _ = draw_text_to_fit(first_name, font, first_name_standardized, font_color, font_thickness, background_color)
    text_img_ln, _ = draw_text_to_fit(last_name, font, last_name_standardized, font_color, font_thickness, background_color)

    # Optionally override with draw_text_centered if you prefer
    # text_img_fn = draw_text_centered(...)
    # text_img_ln = draw_text_centered(...)

    total_width = text_img_fn.shape[1] + fixed_spacing + text_img_ln.shape[1]
    final_img = np.full((standard_height, total_width, 3), background_color, dtype=np.uint8)

    # Place first name
    final_img[:, :text_img_fn.shape[1]] = cv2.resize(text_img_fn, 
                                                     (text_img_fn.shape[1], standard_height))
    # Place last name after spacing
    start_x = text_img_fn.shape[1] + fixed_spacing
    final_img[:, start_x:start_x + text_img_ln.shape[1]] = cv2.resize(text_img_ln, 
                                                                      (text_img_ln.shape[1], standard_height))

    unique_id = str(uuid.uuid4())
    output_filename = f"{gender_par}_{int(time.time())}_{unique_id}.png"
    output_image_path = Path(temp_dir) / output_filename
    cv2.imwrite(str(output_image_path), final_img[:, :, ::-1])  # flip to BGR
    logger.debug(f"Name added to image. Image saved to {output_image_path}")
    logger.info(f"Image saved to {output_image_path}")
    return output_image_path

def add_full_name_to_image(
    name, 
    gender_par, 
    box, 
    font=None, 
    font_size=100, 
    background_color=(0, 0, 0), 
    font_color=(255, 255, 255), 
    font_scale=1, 
    font_thickness=1
):
    """
    Pillow-based reimplementation of add_full_name_to_image.
    """
    StartX, StartY, EndX, EndY = box
    box_width = EndX - StartX

    if font is None:
        font = "arial.ttf"

    text_img, actual_scale = draw_text_to_fit(name, font, box, font_color, font_thickness, background_color)

    # If the text overflows, enlarge
    pil_font = load_font(font, 30)  # approximate
    w, h = measure_text_size_pil(name, pil_font)
    if w > box_width:
        new_width = w + 20
        bigger = np.full((text_img.shape[0], new_width, 3), background_color, dtype=np.uint8)
        bigger[:, :text_img.shape[1]] = text_img
        text_img = bigger

    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"{gender_par}_{int(time.time())}_{unique_id}.png"
    output_image_path = Path(temp_dir) / output_filename

    cv2.imwrite(str(output_image_path), text_img[:, :, ::-1])  # flip RGB->BGR
    logger.info(f"Image saved to {output_image_path}")

    return output_image_path
