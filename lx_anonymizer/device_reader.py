# Device reader module to parse JSON configurations
import json
import cv2
from .box_operations import make_box_from_device_list
from pathlib import Path
from .custom_logger import get_logger

logger=get_logger(__name__)


"""
Functions used for reading the device lists parameters.

These functions will read the device list JSON files and return the parameters.
If the device parameter is set when calling the main function, 
the functions will use the device parameters.

"""


def parse_color(color_str):
    logger.debug("parsing color")
    return tuple(map(int, color_str.strip('()').split(',')))

# Font mapping
FONT_MAP = {
    "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
    "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
    "FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX
}

base_dir = Path(__file__).resolve().parent

def read_device(device):
    logger.debug(f"reading device config for {device}")
    device_file_path = Path(base_dir) / f'devices{device}.json'
    with open(str(device_file_path)) as json_parameters:
        data = json.load(json_parameters)
                    
        background_color = "(255, 255, 255)"
        font_color = "(0, 0, 0)"
        font = "FONT_HERSHEY_DUPLEX"
        font_size = 40
        font_scale = font_size / 20
        font_thickness = 2
        text_formatting = "first_name last_name"
        first_name_x = 0
        first_name_y = 0
        first_name_width = 100
        first_name_height = 20
        last_name_x = 100
        last_name_y = 0
        last_name_width = 100
        last_name_height = 20

        keys_to_check = ["background_color", "text_color", "font", "font_size", "text_formatting", "patient_first_name_x", "patient_first_name_y", "patient_first_name_width", "patient_first_name_height", "patient_last_name_x", "patient_last_name_y", "patient_last_name_width", "patient_last_name_height"]
        for key in data["fields"]:
            if key in keys_to_check:
                if key == "background_color":
                    background_color = parse_color(data["fields"][key])
                elif key == "text_color":
                    font_color = parse_color(data["fields"][key])
                elif key == "font":
                    font_key = data["fields"][key]
                    if font_key in FONT_MAP:
                        font = FONT_MAP[font_key]
                    else:
                        logger.debug(f"Warning: Font '{font_key}' not recognized. Using default font.")
                        font = cv2.FONT_HERSHEY_SIMPLEX
                elif key == "font_size":
                    font_size = data["fields"][key]
                    font_scale = font_size / 20
                    font_thickness = 2
                elif key == "text_formatting":
                    text_formatting = data["fields"][key]
                elif key == "patient_first_name_x":
                    first_name_x = data["fields"][key]
                elif key == "patient_first_name_y":
                    first_name_y = data["fields"][key]
                elif key == "patient_first_name_width":
                    first_name_width = data["fields"][key]
                elif key == "patient_first_name_height":
                    first_name_height = data["fields"][key]
                elif key == "patient_last_name_x":
                    last_name_x = data["fields"][key]
                elif key == "patient_last_name_y":
                    last_name_y = data["fields"][key]
                elif key == "patient_last_name_width":
                    last_name_width = data["fields"][key]
                elif key == "patient_last_name_height":
                    last_name_height = data["fields"][key]

        return background_color, font_color, font, font_scale, font_thickness, text_formatting, first_name_x, first_name_y, first_name_width, first_name_height, last_name_x, last_name_y, last_name_width, last_name_height

def read_name_boxes(device, first_name_x = 0, first_name_y = 0, first_name_width = 100, first_name_height = 20, last_name_x = 100, last_name_y = 0, last_name_width = 100, last_name_height = 20, parameter=False):
    logger.debug(f"reading device patient name config for {device}")
    if parameter==True:
        return None, None
        
    device_file_path = Path(base_dir) / f'devices/{device}.json'
    
    with open(str(device_file_path)) as json_parameters:
        logger.debug(f"device file path opened:{device_file_path}")
        data = json.load(json_parameters)
        
        keys_to_check = ["patient_first_name_x", "patient_first_name_y", "patient_first_name_width", "patient_first_name_height", "patient_last_name_x", "patient_last_name_y", "patient_last_name_width", "patient_last_name_height"]
        for key in data["fields"]:
            if key in keys_to_check:
                if key == "patient_first_name_x":
                    first_name_x = data["fields"][key]
                elif key == "patient_first_name_y":
                    first_name_y = data["fields"][key]
                elif key == "patient_first_name_width":
                    first_name_width = data["fields"][key]
                elif key == "patient_first_name_height":
                    first_name_height = data["fields"][key]
                elif key == "patient_last_name_x":
                    last_name_x = data["fields"][key]
                elif key == "patient_last_name_y":
                    last_name_y = data["fields"][key]
                elif key == "patient_last_name_width":
                    last_name_width = data["fields"][key]
                elif key == "patient_last_name_height":
                    last_name_height = data["fields"][key]
        
        if first_name_x == 0 and first_name_y == 0 and first_name_width == 0 and first_name_height == 0 and last_name_x == 0 and last_name_y == 0 and last_name_width == 0 and last_name_height == 0:
            first_name_box = None
            last_name_box = None
            parameter = True
        else:
            first_name_box = make_box_from_device_list(first_name_x, first_name_y, first_name_width, first_name_height)
            last_name_box = make_box_from_device_list(last_name_x, last_name_y, last_name_width, last_name_height)
        return first_name_box, last_name_box
        
def read_background_color(device):
    device_file_path = Path(base_dir) / f'devices/{device}.json'
    logger.debug(f"reading device background color config for {device}")
    with open(str(device_file_path)) as json_parameters:
        data = json.load(json_parameters)
        
        background_color = "(225, 225, 225)"

        keys_to_check = ["background_color"]
        for key in data["fields"]:
            if key in keys_to_check:
                if key == "background_color":
                    background_color = parse_color(data["fields"][key])
        return background_color
    
def read_text_formatting(device):
    device_file_path =Path(base_dir) / f'devices/{device}.json'
    with open(str(device_file_path)) as json_parameters:
        data = json.load(json_parameters)
        
        text_formatting = "first_name last_name"
        background_color = "(255, 255, 255)"  # Default background color
        font_color = "(0, 0, 0)"  # Default font color
        font = cv2.FONT_HERSHEY_SIMPLEX  # Default font
        font_size = 20
        font_scale = font_size / 20
        font_thickness = 2

        keys_to_check = ["text_formatting", "background_color", "text_color", "font", "font_size", "font_thickness", "font_scale"]
        for key in data["fields"]:
            if key in keys_to_check:
                if key == "text_formatting":
                    text_formatting = data["fields"][key]
                if key == "background_color":
                    background_color = parse_color(data["fields"][key])
                if key == "text_color":
                    font_color = parse_color(data["fields"][key])
                if key == "font":
                    font_key = data["fields"][key]
                    if font_key in FONT_MAP:
                        font = FONT_MAP[font_key]
                    else:
                        logger.debug(f"Warning: Font '{font_key}' not recognized. Using default font.")
                        font = cv2.FONT_HERSHEY_SIMPLEX
                if key == "font_size":
                    font_size = data["fields"][key]
                    font_scale = font_size / 20
                if key == "font_thickness":
                    font_thickness = data["fields"][key]
                if key == "font_scale":
                    font_scale = data["fields"][key]
        return background_color, font_color, font, font_scale, font_thickness, text_formatting