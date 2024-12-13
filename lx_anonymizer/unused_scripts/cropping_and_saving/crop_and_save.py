from PIL import Image
import uuid
import os    


def crop_and_save(image_path, boxes):
    cropped_image_paths = []
    image_texts = {}  # Dictionary to store image paths and their corresponding texts with unique IDs

    with Image.open(image_path) as img:
        for idx, (startX, startY, endX, endY) in enumerate(boxes):
            # Crop the image using the box coordinates
            cropped_img = img.crop((startX, startY, endX, endY))
            
            # Extract the file extension and create a new file name with it
            file_extension = os.path.splitext(image_path)[1]
            cropped_img_name = f"cropped_{idx}{file_extension}"  # Append index and keep original extension
            
            # Ensure the file name is valid
            cropped_img_name = ''.join(c for c in cropped_img_name if c.isalnum() or c in '._-')
            
            # Construct the full path to save the cropped image
            cropped_img_path = os.path.join(image_path, cropped_img_name)

            cropped_img.save(cropped_img_path)
            cropped_image_paths.append(image_path)

            # Generate a unique ID for the extracted text
            unique_id = str(uuid.uuid4())

            # Save the extracted text and its unique ID in the dictionary
            image_texts[unique_id] = {'path': cropped_img_path}

            # Optionally, print or save the image path, text, and unique ID to a file
            print(f"Cropped image saved to: {cropped_img_path}, ID: {unique_id}")

    # Return both the paths and the text with unique IDs
    return cropped_image_paths

