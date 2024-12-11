import os
import shutil
import subprocess
from math import ceil
from PIL import Image

# Resolve all paths relative to the base directory.
BASE_DIR = os.getcwd()

# Define paths relative to the base directory.
VENV_PATH = os.path.join(BASE_DIR, "models/2001_reinhard/2001_reinhard")
PYTHON_EXECUTABLE = os.path.join(VENV_PATH, "bin", "python")
COLOR_TRANSFER_SCRIPT = os.path.join(BASE_DIR, "models/2001_reinhard/color_transfer.py")
TEMP_SOURCE_DIR = os.path.join(BASE_DIR, "models/2001_reinhard/source")
TEMP_TARGET_DIR = os.path.join(BASE_DIR, "models/2001_reinhard/target")
RESULT_DIR = os.path.join(BASE_DIR, "models/2001_reinhard/result")
GRAYSCALE_DIR = os.path.join(BASE_DIR, "data/grayscale")
TRAIN_DIR = os.path.join(BASE_DIR, "data/train")
VAL_DIR = os.path.join(BASE_DIR, "data/val")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/colorized/2001_reinhard")

# Modify color_transfer.py to handle division by 0 from grayscale source images.
def modify_color_transfer(script_path):
    with open(script_path, "r") as file:
        lines = file.readlines()

    # Replace with the mean of the target channel.
    updated_lines = []
    for line in lines:
        if "x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]" in line:
            updated_lines.append("\t\t\t\t\tif s_std[k] == 0:\n")
            updated_lines.append("\t\t\t\t\t\tx = t_mean[k]\n")
            updated_lines.append("\t\t\t\t\telse:\n")
            updated_lines.append("\t" + line)
        else:
            updated_lines.append(line)

    # Write the modified content back to the file.
    with open(script_path, "w") as file:
        file.writelines(updated_lines)

# Convert a JPG image to BMP format.
def convert_to_bmp(source_path, destination_path):
    with Image.open(source_path) as img:
        img.save(destination_path, format="BMP")

# Create a directory structure for the given subfolder.
def create_directory_structure(base_dir, subfolder):
    target_dir = os.path.join(base_dir, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# Batch colorize images from grayscale_dir with reference images from train_dir/val_dir.
def batch_colorize(subfolder, grayscale_dir, train_dir, val_dir, output_dir, batch_size=6):
    grayscale_subfolder = os.path.join(grayscale_dir, subfolder)
    output_subfolder = create_directory_structure(output_dir, subfolder)

    # Get lists of grayscale images and corresponding reference images.
    grayscale_images = sorted(os.listdir(grayscale_subfolder))
    reference_images = [os.path.join(train_dir if "train" in img else val_dir, img.replace("train_", "").replace("val_", "")) for img in grayscale_images]

    num_batches = ceil(len(grayscale_images) / batch_size)

    for batch_num in range(num_batches):
        print(f"Batch {batch_num + 1}/{num_batches} in {subfolder} is being processed.")

        # Select images for this batch.
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(grayscale_images))
        batch_grayscale = grayscale_images[start_idx:end_idx]
        batch_reference = reference_images[start_idx:end_idx]

        # Convert and copy grayscale and reference images to temporary directories.
        for i, (grayscale_img, reference_img) in enumerate(zip(batch_grayscale, batch_reference)):
            convert_to_bmp(os.path.join(grayscale_subfolder, grayscale_img), os.path.join(TEMP_SOURCE_DIR, f"s{i+1}.bmp"))
            convert_to_bmp(reference_img, os.path.join(TEMP_TARGET_DIR, f"t{i+1}.bmp"))

        # Run the color transfer script.
        try:
            subprocess.run(
                [PYTHON_EXECUTABLE, COLOR_TRANSFER_SCRIPT],
                cwd=os.path.join(BASE_DIR, "models/2001_reinhard"),
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error {e} occurred for batch {batch_num + 1}.")

        # Move results to the output directory, preserving batch order.
        for i in range(len(batch_grayscale)):
            result_file = f"r{i+1}.bmp"
            result_path = os.path.join(RESULT_DIR, result_file)

            if not os.path.exists(result_path):
                raise ValueError(f"{result_path} is missing.")

            output_file = batch_grayscale[i]
            shutil.move(
                result_path,
                os.path.join(output_subfolder, output_file),
            )

if __name__ == "__main__":
    modify_color_transfer(COLOR_TRANSFER_SCRIPT)
    subfolders = [f for f in os.listdir(GRAYSCALE_DIR) if os.path.isdir(os.path.join(GRAYSCALE_DIR, f))]
    for subfolder in subfolders:
        batch_colorize(subfolder, GRAYSCALE_DIR, TRAIN_DIR, VAL_DIR, OUTPUT_DIR)
