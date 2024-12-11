import os
import shutil
import subprocess

# Resolve all paths relative to the base directory.
BASE_DIR = os.getcwd()

# Define paths relative to the base directory.
COLORIZATION_DIR = os.path.join(BASE_DIR, "models/2016_zhang")
VENV_PATH = os.path.join(COLORIZATION_DIR, "2016_zhang")
PYTHON_EXECUTABLE = os.path.join(VENV_PATH, "bin", "python")
COLORIZATION_SCRIPT = os.path.join(COLORIZATION_DIR, "demo_release.py")
GRAYSCALE_DIR = os.path.join(BASE_DIR, "data/grayscale")
TRAIN_DIR = os.path.join(BASE_DIR, "data/train")
VAL_DIR = os.path.join(BASE_DIR, "data/val")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/colorized/2016_zhang")

# Modify the script for I/O naming and removing plotting.
def modify_colorization_script(script_path):
    with open(script_path, "r") as file:
        lines = file.readlines()

    updated_lines = []

    for line in lines:
        stripped_line = line.lstrip()

        # Remove SIGGRAPH17 model.
        if "siggraph17" in stripped_line:
            continue

        # Remove plotting commands except plt.imsave.
        if stripped_line.startswith('plt'):
            if stripped_line.startswith('plt.imsave'):
                # Modify plt.imsave to overwrite the input image.
                new_line = "plt.imsave(opt.img_path, out_img_eccv16)\n"
                updated_lines.append(new_line)
            else:
                continue
        else:
            updated_lines.append(line)

    # Write the content back to the file.
    with open(script_path, "w") as file:
        file.writelines(updated_lines)

# Create a directory structure for the given subfolder.
def create_directory_structure(base_dir, subfolder):
    target_dir = os.path.join(base_dir, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# Colorize images.
def colorize(subfolder, grayscale_dir, output_dir):
    grayscale_subfolder = os.path.join(grayscale_dir, subfolder)
    output_subfolder = create_directory_structure(output_dir, subfolder)

    # Get lists of grayscale images.
    grayscale_images = sorted(os.listdir(grayscale_subfolder))

    for i in range(len(grayscale_images)):
        grayscale = grayscale_images[i]

        # Copy grayscale image to COLORIZATION_DIR.
        shutil.copy2(os.path.join(grayscale_subfolder, grayscale), os.path.join(COLORIZATION_DIR, os.path.basename(grayscale)))

        # Run the colorization script.
        try:
            subprocess.run(
                [PYTHON_EXECUTABLE, COLORIZATION_SCRIPT, "--img_path", grayscale],
                cwd=os.path.join(BASE_DIR, "models/2016_zhang"),
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error {e} occurred for image {grayscale}.")

        # Move result to the output directory.
        colorized_image_path = os.path.join(COLORIZATION_DIR, grayscale)
        shutil.move(colorized_image_path, os.path.join(output_subfolder, grayscale))

if __name__ == "__main__":
    modify_colorization_script(COLORIZATION_SCRIPT)
    subfolders = [f for f in os.listdir(GRAYSCALE_DIR) if os.path.isdir(os.path.join(GRAYSCALE_DIR, f))]
    for subfolder in subfolders:
        colorize(subfolder, GRAYSCALE_DIR, OUTPUT_DIR)
