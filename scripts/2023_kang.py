import os
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Define paths.
BASE_DIR = os.getcwd()
GRAYSCALE_DIR = os.path.join(BASE_DIR, "data/grayscale")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/colorized/2023_kang")

# Ensure the output directory exists.
def create_directory_structure(base_dir, subfolder):
    target_dir = os.path.join(base_dir, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# Run colorization directly.
def colorize_images(subfolder, grayscale_dir, output_dir):
    grayscale_subfolder = os.path.join(grayscale_dir, subfolder)
    output_subfolder = create_directory_structure(output_dir, subfolder)

    # Get grayscale images.
    grayscale_images = sorted(os.listdir(grayscale_subfolder))

    # Initialize the pipeline once to avoid repeated initialization.
    img_colorization = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')

    for grayscale in grayscale_images:
        grayscale_image_path = os.path.join(grayscale_subfolder, grayscale)
        colorized_image_path = os.path.join(output_subfolder, grayscale)

        # Skip if colorized image already exists.
        if os.path.exists(colorized_image_path):
            continue

        try:
            # Perform image colorization.
            result = img_colorization(grayscale_image_path)
            cv2.imwrite(colorized_image_path, result[OutputKeys.OUTPUT_IMG])
        except Exception as e:
            print(f"Error {e} occurred processing {grayscale}.")

# Define main function.
if __name__ == "__main__":
    subfolders = [f for f in os.listdir(GRAYSCALE_DIR) if os.path.isdir(os.path.join(GRAYSCALE_DIR, f))]
    for subfolder in subfolders:
        colorize_images(subfolder, GRAYSCALE_DIR, OUTPUT_DIR)
