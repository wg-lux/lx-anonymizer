# AGL Anonymizer Pipeline

This Module is designed to work with the Django API AGL Anonymizer.


The Submodule is used to provide automatic anonymization of sensitive information. It is a three step pipeline using Text Region Detection (EAST), OCR (Optical Character Recognition, Tesseract, TrOCR) and Named Entity Recognition (flair-ner, gender-guessr). This tool is particularly useful in scenarios where sensitive information needs to be removed or replaced from images or documents while retaining the overall context and visual structure.

##

For testing abd development:

run

nix build
nix develop

use the pipeline by providing:

python -m agl_anonymizer_pipeline.main -i /your_image.png -d olympus_cv_1500 -c 0.5 -w 320 -e 320

## Features

- **Text detection and anonymization**: Utilizes advanced OCR techniques to detect text in images and applies anonymizing to safeguard sensitive information.
- **Blurring Functionality**: Offers customizable blurring options to obscure parts of an image, providing an additional layer of privacy.
- **Extensive Format Support**: Capable of handling various image and document formats for a wide range of applications.

## Installation

To get started with AGL Anonymizer, clone this repository and install the required dependencies.

git clone https://github.com/maxhild/agl_anonymizer_pipeline.git
cd agl_anonymizer_pipeline
nix develop
dowload a text detection model like frozen_east_text_detection.pb and place it inside the agl_anonymizer_pipeline folder.


## Usage

To use AGL Anonymizer Pipeline, follow these steps:

Prepare Your Images: Place the images you want to process in the designated folder.
Configure Settings: Adjust the settings in the configuration file (if applicable) to suit your anonymizing and blurring needs.
Run the Module: Execute the main script from the command line to process the images.
bash

code:

python main.py --image images/your_image.jpg --east frozen_east_text_detection.pb 

example:

python main.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb 

## Modules

AGL Anonymizer is comprised of several key modules:

Text Region Detection: EAST and Tesseract are applied to find the text regions in the image.
OCR Module: Detects and extracts text from images using Tesseract and TROCR
Anonymizer Module: Applies Pseudonyms to the sensitive text regions identified. A custom names directory is provided.
Blur Module: Provides functions to blur specific areas in the image.
Save Module: Handles the saving of processed images in a chosen format.

## Contributing

Contributions to the AGL Anonymizer Pipeline are welcome! If you have suggestions for improvements or bug fixes, please open an issue or a pull request.

TO DO:

- UTF-8 Handling of Names - cv2 putText only works in ASCII
- Improving the text region detection by Model Training
- You can customize the behavior of AGL Anonymizer by modifying the parameters in the config.py file (if included). This includes adjusting the OCR sensitivity, blur intensity, and more.


## License

This project is licensed under the MIT License.

## Contact

For any inquiries or assistance with AGL Anonymizer, please contact Max Hild at Maxhild10@gmail.com.

# AGL Anonymizer Pipeline

This Module is designed to work with the Django API AGL Anonymizer.


The Submodule is used to provide automatic anonymization of sensitive information. It is a three step pipeline using Text Region Detection (EAST), OCR (Optical Character Recognition, Tesseract, TrOCR) and Named Entity Recognition (flair-ner, gender-guessr). This tool is particularly useful in scenarios where sensitive information needs to be removed or replaced from images or documents while retaining the overall context and visual structure.


## Contributing

Contributions to AGL Anonymizer are welcome! If you have suggestions for improvements or bug fixes, please open an issue or a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or assistance with AGL Anonymizer, please contact Max Hild at Maxhild10@gmail.com.


## Installation

To get started with AGL anonymizer, clone this repository and install the required dependencies. Nix and Poetry should install the dependencies automatically.

The package is also available on pip through:

pip install agl_anonymizer_pipeline

git clone https://github.com/wg-lux/agl_anonymizer_pipeline.git

## Usage

To use AGL anonymizer, follow these steps:

Prepare Your Images: Place the images you want to process in the designated folder.
Configure Settings: Adjust the settings in the configuration file (if applicable) to suit your anonymizing and blurring needs.
Run the Module: Execute the main script to process the images.

```bash
python main.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb 
```

## Parameters of the `main` function

The `main` function is responsible for processing either images or PDF files through the AGL Anonymizer pipeline. Below are the parameters it accepts:

- **image_or_pdf_path** (`str`):  
   The path to the input image or PDF that you want to process. This can be a single image file or a multi-page PDF. The function will detect the file type and process accordingly.

- **east_path** (`str`, optional):  
   Path to the pre-trained EAST text detection model (`frozen_east_text_detection.pb`). If not provided, the function will expect it to be in the designated location in the AGL Anonymizer setup.

- **device** (`str`, optional):  
   The device name used to set the correct OCR and NER (Named Entity Recognition) text settings for different devices. Defaults to `olympus_cv_1500`.

- **validation** (`bool`, optional):  
   If set to `True`, the function will perform additional validation by using an external lx-annotate service to validate the results and return extra output. Defaults to `False`.

- **min_confidence** (`float`, optional):  
   Minimum confidence level for detecting text regions within the image. Regions with a confidence score below this threshold will not be processed. Defaults to `0.5`.

- **width** (`int`, optional):  
   Resized width for the image, used for text detection. It should be a multiple of 32. Defaults to `320`.

- **height** (`int`, optional):  
   Resized height for the image, used for text detection. It should be a multiple of 32. Defaults to `320`.

### Example usage of the `main` function:
```python
main(
    image_or_pdf_path="path/to/your/file.pdf",
    east_path="path/to/frozen_east_text_detection.pb",
    device="olympus_cv_1500",
    validation=True,
    min_confidence=0.6,
    width=640,
    height=640
)
```

# Modules

AGL Anonymizer is comprised of several key modules:

## Directory Setup Module

The **Directory Setup Module** defines and manages the storage locations for anonymized data, intermediate results, and other important files used during the anonymization process. It includes functions for creating the main, temporary, and blur directories used in the anonymization pipeline.

### Key Components

1. **Main Directory**:
   - Stores the final anonymized results, structured study data, and training data ready for export.
   - Managed by the `create_main_directory` function.

2. **Temporary Directory**:
   - Used to store intermediate results during the anonymization process.
   - Cleaned up regularly to prevent clutter.
   - Managed by the `create_temp_directory` function.

3. **Blur Directory**:
   - Stores blurred images generated during the anonymization process.
   - Cleaned up regularly.
   - Managed by the `create_blur_directory` function.

### Functions

#### `create_directories(directories)`

- **Purpose**: A helper function that creates a list of directories if they do not already exist.
- **Parameters**:
  - **`directories`** (`list`): A list of directory paths to create.
- **Functionality**:
  - Iterates over each directory path in the list.
  - Checks if the directory exists; if not, creates it.
  - Logs a message if the directory already exists or if it was successfully created.
  - If an error occurs, logs the error and raises an exception.
- **Returns**: None.

#### `create_main_directory(default_main_directory)`

- **Purpose**: Creates the main directory in a writable location, typically outside the Nix store.
- **Parameters**:
  - **`default_main_directory`** (`str`): The base directory path where the main directory will be created.
- **Functionality**:
  - If the directory does not exist, attempts to create it.
  - Logs messages indicating whether the directory already exists or if it was created successfully.
  - If an error occurs during the creation, logs the error and raises an exception.
- **Returns**: The path to the main directory.

#### `create_temp_directory(default_temp_directory, default_main_directory)`

- **Purpose**: Creates both the 'temp' directory and the 'csv_training_data' directory used for intermediate results and training data during the anonymization process.
- **Parameters**:
  - **`default_temp_directory`** (`str`): The path where the temp directory will be created.
  - **`default_main_directory`** (`str`): The path to the main directory, where the csv directory will be created.
- **Functionality**:
  - Creates a 'temp' directory in the temporary directory and a 'csv_training_data' directory in the main directory.
  - Logs whether the directories already exist or were created successfully.
  - Returns the paths to the temp directory, main directory, and csv directory.
- **Returns**: A tuple containing paths to the temp directory, main directory, and csv directory.

#### `create_blur_directory(default_main_directory)`

- **Purpose**: Creates a 'blurred_images' directory in the main directory to store images that have been blurred during the anonymization process.
- **Parameters**:
  - **`default_main_directory`** (`str`): The path where the blurred images directory will be created.
- **Functionality**:
  - If the directory does not exist, attempts to create it.
  - Logs whether the directory already exists or was created successfully.
  - If an error occurs during the creation, logs the error and raises an exception.
- **Returns**: The path to the 'blurred_images' directory.

## OCR Pipeline Manager Module

The **OCR Pipeline Manager** module coordinates the Optical Character Recognition (OCR) and Named Entity Recognition (NER) processes for images and PDFs. It uses multiple OCR techniques (such as Tesseract and TrOCR), applies NER for detecting sensitive information, and replaces detected names with pseudonyms using the names generator. This module is essential for extracting and anonymizing text from input files.

### Key Components

1. **OCR and NER Functions**:
   - **`trocr_on_boxes(img_path, boxes)`**:  
     Uses the TrOCR model for OCR on specific regions (boxes) in the image.
   - **`tesseract_on_boxes(img_path, boxes)`**:  
     Uses Tesseract OCR for detecting text within the provided boxes.
   - **`NER_German(text)`**:  
     Applies Named Entity Recognition (NER) on the extracted text to identify entities such as names (tagged as `PER` for persons).

2. **Text Detection**:
   - **`east_text_detection(img_path, east_path, min_confidence, width, height)`**:  
     Uses the EAST text detection model to identify potential text regions in the image.
   - **`tesseract_text_detection(img_path, min_confidence, width, height)`**:  
     Uses Tesseract's built-in text detection to identify text regions in the image.

3. **Name Handling**:
   - **`gender_and_handle_full_names(words, box, image_path, device)`**:  
     Replaces full names in detected text with pseudonyms based on gender predictions.
   - **`gender_and_handle_separate_names(words, first_name_box, last_name_box, image_path, device)`**:  
     Handles cases where first and last names are detected separately in the image.
   - **`gender_and_handle_device_names(words, box, image_path, device)`**:  
     Handles the names associated with specific medical devices.

4. **Image Processing**:
   - **`blur_function(image_path, box, background_color)`**:  
     Blurs specific regions (text boxes) in an image, typically for anonymization.
   - **`convert_pdf_to_images(pdf_path)`**:  
     Converts a PDF document into individual images for processing.

5. **Combining Text Boxes**:
   - **`combine_boxes(text_with_boxes)`**:  
     Merges adjacent text boxes if they belong to the same line and are close together.

6. **Helper Functions**:
   - **`find_or_create_close_box(phrase_box, boxes, image_width, offset=60)`**:  
     Finds or creates a bounding box that is close to the existing text box, useful when handling names or phrases that may extend beyond the detected region.
   - **`process_text(extracted_text)`**:  
     Cleans up extracted text, removing excess line breaks and spaces.

### Main Functionality

#### `process_images_with_OCR_and_NER(file_path, east_path, device, min_confidence, width, height)`

This is the core function of the module, which handles the entire OCR and NER pipeline for a given file (image or PDF). It performs the following steps:
- Detects and reads text from the file using EAST and Tesseract models.
- Applies OCR (TrOCR and Tesseract) to the detected text regions.
- Uses NER to identify sensitive information (e.g., names) in the text.
- Replaces detected names with pseudonyms using the names generator.
- Optionally blurs specified regions in the image, such as detected names.
- Outputs a modified version of the image with anonymized text and a CSV file containing the NER results.

##### Parameters:
- **`file_path`** (`str`): The path to the input image or PDF file.
- **`east_path`** (`str`, optional): The path to the EAST model used for text detection.
- **`device`** (`str`, optional): Specifies the device configuration for text handling and name pseudonymization. Defaults to `"default"`.
- **`min_confidence`** (`float`, optional): The minimum confidence level required for text detection. Defaults to `0.5`.
- **`width`** (`int`, optional): The width to resize the image for text detection. Defaults to `320`.
- **`height`** (`int`, optional): The height to resize the image for text detection. Defaults to `320`.

##### Returns:
- **`modified_images_map`** (`dict`): A map of the modified images with replaced text.
- **`result`** (`dict`): Contains detailed results of the OCR and NER processes, including:
  - `filename`: The original file name.
  - `file_type`: The type of the file (image or PDF).
  - `extracted_text`: The raw extracted text from the file.
  - `names_detected`: A list of detected names.
  - `combined_results`: The OCR and NER results.
  - `modified_images_map`: A map of modified images with pseudonymized text.
  - `gender_pars`: List of gender classifications used in the pseudonymization process.

### Example Usage

```python
file_path = "your_file_path.jpg"
modified_images_map, result = process_images_with_OCR_and_NER(
    file_path, 
    east_path="path/to/frozen_east_text_detection.pb",
    device="default", 
    min_confidence=0.6, 
    width=640, 
    height=640
)
for res in result['combined_results']:
    print(res)
```

## Text Detection Module

The **Text Detection Module** is responsible for detecting text regions within an image using two primary methods:

1. **EAST Text Detection**: A pre-trained deep learning model that detects text regions in an image using OpenCV.
2. **Tesseract OCR Text Detection**: Uses Tesseract OCR to detect and extract text from an image.

Both methods can be used as part of an anonymization pipeline to detect text regions for further processing (e.g., blurring, pseudonymization).

### Key Components

1. **EAST Text Detection**:
   - **`east_text_detection(image_path, east_path, min_confidence, width, height)`**:
     - Uses the EAST model to detect text regions in an image.
     - Applies non-max suppression to remove overlapping boxes.
     - Returns bounding boxes and confidence scores in JSON format.

2. **Tesseract Text Detection**:
   - **`tesseract_text_detection(image_path, min_confidence, width, height)`**:
     - Uses Tesseract OCR to detect text and corresponding bounding boxes in an image.
     - Returns bounding boxes and confidence scores in JSON format.

3. **Helper Functions**:
   - **`sort_boxes(boxes)`**: 
     - Sorts the detected bounding boxes by their Y-coordinate and X-coordinate to organize the boxes that appear on the same line.
   - **`extend_boxes_if_needed(image, boxes)`**:
     - Extends bounding boxes if necessary (not provided but referenced in code). This would be useful for cases where the detected text boxes are too small.

### Main Functionality

#### `east_text_detection(image_path, east_path=None, min_confidence=0.5, width=320, height=320)`

- **Purpose**: Detects text regions in an image using the EAST model, which is ideal for detecting text areas in scenes.
- **Parameters**:
  - **`image_path`** (`str`): Path to the input image file.
  - **`east_path`** (`str`, optional): Path to the EAST model. If not provided, a pre-downloaded model path is used.
  - **`min_confidence`** (`float`, optional): Minimum confidence level required to consider a detection valid. Defaults to `0.5`.
  - **`width`** (`int`, optional): Width to resize the image for the model. Defaults to `320`.
  - **`height`** (`int`, optional): Height to resize the image for the model. Defaults to `320`.
- **Functionality**:
  - Resizes the image to a predefined width and height.
  - Passes the image through the EAST model to detect text regions.
  - Extracts and scales the bounding boxes and confidence scores.
  - Applies non-maxima suppression to remove redundant or overlapping bounding boxes.
  - Returns bounding boxes and their respective confidence scores.
- **Returns**:
  - **`output_boxes`** (`list` of tuples): List of bounding boxes in the format `(startX, startY, endX, endY)`.
  - **`confidences`** (`str`): JSON string of the confidence scores for each bounding box.

#### `tesseract_text_detection(image_path, min_confidence=0.5, width=320, height=320)`

- **Purpose**: Detects text regions in an image using Tesseract OCR. Unlike EAST, this method directly extracts the text as well as the bounding boxes.
- **Parameters**:
  - **`image_path`** (`str`): Path to the input image file.
  - **`min_confidence`** (`float`, optional): Minimum confidence level required to consider a detection valid. Defaults to `0.5`.
  - **`width`** (`int`, optional): Width to resize the image for the model. Defaults to `320`.
  - **`height`** (`int`, optional): Height to resize the image for the model. Defaults to `320`.
- **Functionality**:
  - Resizes the image to a predefined width and height.
  - Passes the image through Tesseract OCR to detect text regions and extract the text content.
  - Extracts bounding boxes and confidence scores for each detected text region.
  - Returns bounding boxes and their respective confidence scores in JSON format.
- **Returns**:
  - **`output_boxes`** (`list` of tuples): List of bounding boxes in the format `(startX, startY, endX, endY)`.
  - **`confidences`** (`str`): JSON string of the confidence scores for each bounding box.

### Helper Function Details

#### `sort_boxes(boxes)`

- **Purpose**: Sorts bounding boxes based on their Y-coordinate and X-coordinate to maintain a consistent order (e.g., reading order).
- **Parameters**:
  - **`boxes`** (`list` of tuples): List of bounding boxes in the format `(startX, startY, endX, endY)`.
- **Returns**: Sorted list of bounding boxes based on their position.

#### `extend_boxes_if_needed(image, boxes)`

- **Purpose**: Extends the detected text boxes if they are too small. This function is useful for better text region detection and visual display.
- **Parameters**:
  - **`image`** (`numpy array`): The image on which text detection was performed.
  - **`boxes`** (`list` of tuples): List of bounding boxes in the format `(startX, startY, endX, endY)`.
- **Returns**: Extended bounding boxes to improve text region detection.

### Example Usage

```python
from text_detection import east_text_detection, tesseract_text_detection

# Example for EAST text detection
image_path = "path/to/your/image.jpg"
east_boxes, east_confidences = east_text_detection(image_path, min_confidence=0.6, width=640, height=640)
print("EAST Detected Boxes:", east_boxes)
print("EAST Confidence Scores:", east_confidences)

# Example for Tesseract text detection
tesseract_boxes, tesseract_confidences = tesseract_text_detection(image_path, min_confidence=0.6, width=640, height=640)
print("Tesseract Detected Boxes:", tesseract_boxes)
print("Tesseract Confidence Scores:", tesseract_confidences)
```

## OCR Module

The **OCR Module** provides functions to perform Optical Character Recognition (OCR) on specified regions within images using two different OCR engines:

1. **TrOCR** (Transformer-based OCR) from Microsoft's Hugging Face models.
2. **Tesseract OCR**, an open-source OCR engine.

This module allows for efficient and accurate text extraction from predefined bounding boxes in images, which is essential in the process of detecting and anonymizing sensitive information.

### Key Components

1. **Model Preloading**:
   - **`preload_models()`**:
     - Preloads the necessary models and tokenizer for TrOCR.
     - Loads the `ViTImageProcessor`, `VisionEncoderDecoderModel`, and `AutoTokenizer` from Hugging Face.
     - Determines the computation device (GPU if available, else CPU).
     - Returns the loaded processor, model, tokenizer, and device for use in OCR functions.

2. **OCR Functions**:
   - **`trocr_on_boxes(image_path, boxes)`**:
     - Performs OCR on specified bounding boxes within an image using the TrOCR model.
     - Utilizes the `expand_roi` function to slightly expand each bounding box for better OCR accuracy.
     - Processes each box individually, converting the cropped image to tensors, and runs it through the model to get the predicted text.
     - Calculates a confidence score based on the model's output scores.
     - Returns a list of extracted texts with their corresponding expanded boxes and confidence scores.
   - **`tesseract_on_boxes(image_path, boxes)`**:
     - Performs OCR on specified bounding boxes within an image using Tesseract OCR.
     - Also uses the `expand_roi` function to expand bounding boxes.
     - Processes each box by cropping and passing it to Tesseract for OCR.
     - Retrieves confidence scores from Tesseract's output data.
     - Returns a list of extracted texts with their corresponding expanded boxes and confidence scores.

3. **Helper Functions**:
   - **`expand_roi(startX, startY, endX, endY, padding, image_shape)`**:
     - Expands the Region of Interest (ROI) by a specified padding while ensuring the new ROI stays within the image boundaries.
     - Helps in capturing more context around the text, improving OCR accuracy.

### Main Functionality

#### `preload_models()`

- **Purpose**: Preloads the TrOCR models and tokenizer for efficient reuse, especially when processing multiple images or boxes.
- **Functionality**:
  - Checks for available computation devices (GPU or CPU).
  - Loads the image processor, model, and tokenizer from the `microsoft/trocr-base-str` pre-trained model.
  - Moves the model to the selected device for computation.
- **Returns**: `processor`, `model`, `tokenizer`, `device`.

#### `trocr_on_boxes(image_path, boxes)`

- **Purpose**: Performs OCR using the TrOCR model on specified bounding boxes in an image.
- **Parameters**:
  - **`image_path`** (`str`): Path to the image file.
  - **`boxes`** (`list` of tuples): A list of bounding boxes, where each box is represented as `(startX, startY, endX, endY)`.
- **Functionality**:
  - Loads and converts the image to RGB format.
  - Iterates over each box:
    - Expands the box using `expand_roi` to include a margin around the text.
    - Crops the image to the expanded box.
    - Processes the cropped image using the preloaded processor.
    - Generates text predictions using the model.
    - Calculates a confidence score from the model's output scores.
  - Collects the extracted text and confidence scores.
- **Returns**:
  - **`extracted_text_with_boxes`** (`list`): A list of tuples containing the extracted text and corresponding expanded boxes.
  - **`confidences`** (`list`): A list of confidence scores for each extracted text.

#### `tesseract_on_boxes(image_path, boxes)`

- **Purpose**: Performs OCR using Tesseract on specified bounding boxes in an image.
- **Parameters**:
  - Same as `trocr_on_boxes`.
- **Functionality**:
  - Similar to `trocr_on_boxes`, but uses Tesseract OCR for text extraction.
  - Retrieves detailed OCR data to compute confidence scores.
- **Returns**:
  - Same as `trocr_on_boxes`.

### Example Usage

```python
from ocr_module import preload_models, trocr_on_boxes, tesseract_on_boxes

# Preload models once when the script runs
processor, model, tokenizer, device = preload_models()

# Define your image path and bounding boxes
image_path = "path/to/your/image.jpg"
boxes = [
    (50, 50, 200, 150),
    (250, 80, 400, 180),
    # Add more boxes as needed
]

# Perform OCR using TrOCR
trocr_results, trocr_confidences = trocr_on_boxes(image_path, boxes)
print("TrOCR Results:", trocr_results)
print("TrOCR Confidences:", trocr_confidences)

# Perform OCR using Tesseract
tesseract_results, tesseract_confidences = tesseract_on_boxes(image_path, boxes)
print("Tesseract Results:", tesseract_results)
print("Tesseract Confidences:", tesseract_confidences)
```


## Names Generator Module

The Names Generator module is responsible for assigning gender-specific or neutral names to detected text boxes in images. It uses the gender guesser tool to determine the likely gender of a first name and then selects an appropriate full name from predefined lists of male, female, and neutral names. This process enhances the anonymization workflow by replacing potentially sensitive text with randomized names while preserving the gender or neutrality of the original text.

### Key Components

1. **Gender Guesser**:  
   This tool predicts the gender of the given first name using the `gender_guesser.detector` module. Based on the first name, it returns one of the following values:
   - `male`
   - `mostly_male`
   - `female`
   - `mostly_female`
   - `unknown`
   - `andy` (for androgynous names)

2. **Name Files**:  
   The module uses text files containing ASCII-formatted first and last names:
   - `first_and_last_names_female_ascii.txt`
   - `first_and_last_names_male_ascii.txt`
   - `first_names_female_ascii.txt`
   - `last_names_female_ascii.txt`
   - `first_names_male_ascii.txt`
   - `last_names_male_ascii.txt`
   - `first_names_neutral_ascii.txt`
   - `last_names_neutral_ascii.txt`

   These files are loaded during initialization to provide randomized name selection based on gender.

3. **Functions**:
   - **`gender_and_handle_full_names(words, box, image_path, device)`**:  
     This function handles full names extracted from the image. It determines the gender from the first name and then selects a random full name (first and last) from the appropriate list.
     - Input: Words list, bounding box of text, image path.
     - Output: Processed image path with a pseudonymized name added and the guessed gender.
   
   - **`gender_and_handle_separate_names(words, first_name_box, last_name_box, image_path, device)`**:  
     This function deals with cases where first and last names are separated in the image. It processes the names individually, applies gender guessing, and adds the pseudonymized name to the image.
     - Input: Words list, bounding boxes for first and last name, image path.
     - Output: Processed image path with pseudonymized separate names and the guessed gender.

   - **`gender_and_handle_device_names(words, box, image_path, device)`**:  
     This function works similarly to the full names handler but focuses on names generated by specific medical devices.
     - Input: Words list, bounding box of text, image path, and device name.
     - Output: Image with pseudonymized names based on device-specific rules.

4. **Name Formatting**:
   - **`format_name(name, format_string)`**:  
     This function formats the given name based on the specified device’s formatting rules. For example, it can reorder first and last names depending on the requirements.

5. **Text Rendering**:
   - The module provides several functions for drawing and fitting text to an image. The drawn names are centered, scaled, and resized to fit within the given bounding boxes. Examples include:
     - **`draw_text_with_line_break`**: Renders text with line breaks.
     - **`draw_text_without_line_break`**: Renders text without line breaks.
     - **`draw_text_to_fit`**: Scales and positions text to fit inside a bounding box.

### Example Usage

The names generator module is integrated into the anonymization pipeline and automatically handles the detection and replacement of names in images. Here’s how it can be invoked programmatically:

```python
box_to_image_map, gender_guess = gender_and_handle_full_names(
    words=["John", "Doe"],
    box=(50, 50, 300, 100),
    image_path="path/to/image.png",
    device="olympus_cv_1500"
)
```

## Box Operations Module

The **Box Operations Module** provides various utilities for managing bounding boxes in an image, including expanding, creating, and extending boxes, as well as handling color-based region analysis. This module is particularly useful for detecting and manipulating regions of interest (ROIs) in an image during text detection and image anonymization processes.

### Key Components

1. **Creating and Manipulating Bounding Boxes**:
   - **`make_box_from_name(image, name, padding=10)`**:
     - Creates a bounding box around a given name in an image.
     - Uses the size of the name text and padding to determine the bounding box dimensions.
     - Returns the bounding box coordinates as a tuple `(startX, startY, endX, endY)`.

   - **`make_box_from_device_list(x, y, w, h)`**:
     - Generates a bounding box based on the given coordinates `(x, y)` and size `(w, h)`, which can be used in OpenCV for drawing rectangles.
     - Returns the bounding box as `(startX, startY, endX, endY)`.

2. **Expanding and Extending Bounding Boxes**:
   - **`expand_roi(startX, startY, endX, endY, expansion, image_shape)`**:
     - Expands a bounding box by a specified number of pixels in all directions while ensuring that the box remains within the bounds of the image.
     - Parameters:
       - `startX`, `startY`, `endX`, `endY`: Coordinates of the ROI.
       - `expansion`: Number of pixels to expand the box.
       - `image_shape`: Shape of the image to ensure the box remains within bounds.
     - Returns the expanded bounding box as `(startX, startY, endX, endY)`.

   - **`extend_boxes_if_needed(image, boxes, extension_margin=10, color_threshold=30)`**:
     - Extends the bounding box in different directions based on the color contrast between the current box and adjacent regions.
     - This ensures that the box properly covers the desired region (e.g., a name) even if part of the name extends beyond the box.
     - Parameters:
       - `image`: Input image (NumPy array).
       - `boxes`: List of bounding boxes.
       - `extension_margin`: Number of pixels to extend the box if needed (default: 10).
       - `color_threshold`: Minimum color difference required to extend the box (default: 30).
     - Returns the list of extended bounding boxes.

3. **Analyzing Regions**:
   - **`get_dominant_color(image, box)`**:
     - Analyzes a region within the image (defined by the bounding box) and calculates the dominant color in that region.
     - Uses the k-means clustering algorithm to identify the most common color in the region.
     - Parameters:
       - `image`: Input image (NumPy array).
       - `box`: Bounding box for the region `(startX, startY, endX, endY)`.
     - Returns the dominant color as a tuple `(B, G, R)`.

4. **Combining OCR Results**:
   - **`create_combined_phrases(ocr_texts_with_boxes)`**:
     - Combines multiple OCR-detected phrases into a single phrase if the boxes are contiguous.
     - Merges the bounding boxes of the combined phrases into a single bounding box.
     - Parameters:
       - `ocr_texts_with_boxes`: List of tuples containing OCR-detected text and corresponding bounding boxes.
     - Returns a list of combined phrases and their merged bounding boxes.

5. **Reassembling Modified Images**:
   - **`reassemble_image(modified_images_map, output_dir, id, original_image_path=None)`**:
     - Reassembles an image by overlaying modified regions (e.g., anonymized text) onto the original image.
     - This is typically used to reconstruct an image after certain regions have been modified.
     - Parameters:
       - `modified_images_map`: A dictionary mapping the box coordinates and original image path to the modified image path.
       - `output_dir`: Directory where the final reassembled image should be saved.
       - `id`: A unique identifier for the reassembled image.
       - `original_image_path`: Path to the original image to be reassembled.
     - Returns the path to the final reassembled image.

### Example Usage

1. **Expanding a Region of Interest (ROI)**:
   ```python
   image_shape = (1080, 1920)  # Example image shape (height, width)
   expanded_box = expand_roi(100, 100, 200, 200, 10, image_shape)
   print(f"Expanded Box: {expanded_box}")
   ```

2. **Creating a Box Around a Name**:
```python
image = cv2.imread('example_image.jpg')
name = "John Doe"
box = make_box_from_name(image, name, padding=15)
print(f"Bounding Box for Name: {box}")
3. **Reassembling an Image with Modifications**:
```

```python
modified_images_map = {
    (('100,100,200,200', 'original_image.jpg'), 'modified_image1.jpg'): 'modified_image1.jpg',
    (('150,150,250,250', 'original_image.jpg'), 'modified_image2.jpg'): 'modified_image2.jpg'
}
output_dir = 'output_directory'
id = 'example_image_id'
reassembled_image_path = reassemble_image(modified_images_map, output_dir, id, original_image_path='original_image.jpg')
print(f"Reassembled Image Saved At: {reassembled_image_path}")
```

4. **Extending Bounding Boxes Based on Color Contrast**:

```python
image = cv2.imread('example_image.jpg')
boxes = [(100, 100, 200, 200), (150, 150, 250, 250)]
extended_boxes = extend_boxes_if_needed(image, boxes, extension_margin=10, color_threshold=20)
print(f"Extended Boxes: {extended_boxes}")
```

### Error Handling
The functions in this module ensure that bounding boxes are correctly formatted before processing.
If a bounding box is not in the correct format (e.g., not a tuple or list of four elements), the functions raise a ValueError to prevent unexpected behavior.
In the reassemble_image function, warnings are printed if the original or modified images cannot be loaded. This ensures the program can continue running without crashing.
Performance Considerations
Color-Based Operations: Functions like get_dominant_color and extend_boxes_if_needed perform k-means clustering and color analysis. These operations can be computationally expensive on large images or regions. Consider optimizing by using smaller regions or downscaled images for color analysis.
Resizing and Bounding Box Calculations: Ensure that bounding box coordinates are correctly scaled to the dimensions of the image when applying transformations or overlays.
Dependencies
OpenCV: Used for image processing and drawing bounding boxes.
NumPy: Used for array manipulations and color analysis.
UUID: Used for generating unique identifiers for reassembled images.
K-means Clustering: Used in get_dominant_color for color analysis.
Integration
This module is typically used in conjunction with OCR pipelines or anonymization processes where text regions need to be detected, processed, and modified within images. The functions are designed to handle box-based operations and modifications efficiently.