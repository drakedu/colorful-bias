import os
import subprocess
import sys

# Define models and their repositories.
models = [
    {
        "name": "2001 Reinhard",
        "repo": "https://github.com/chia56028/Color-Transfer-between-Images",
    },
]

# Create base folders.
folders = ["data/colorized", "models"]

def create_folders():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def set_up():
    for model in models:
        try:
            # Create safe name version of model name.
            model['safe_name'] = model['name'].replace(" ", "_").lower()

            # Create a folder for the colorized images.
            colorized_path = os.path.join("data", "colorized", model["safe_name"])
            os.makedirs(colorized_path, exist_ok=True)

            # Create a folder for the model.
            model_path = os.path.join("models", model["safe_name"])
            if not os.path.exists(model_path):
                subprocess.run(["git", "clone", model["repo"], model_path], check=True)

            # Create a virtual environment.
            env_path = os.path.join(model_path, model["safe_name"])
            if not os.path.exists(env_path):
                subprocess.run([sys.executable, "-m", "venv", env_path], check=True)

            # Install dependencies.
            requirements_file = os.path.join("requirements", model["safe_name"] + ".txt")
            if os.path.exists(requirements_file):
                pip_path = os.path.join(env_path, "bin", "pip")
                subprocess.run([pip_path, "install", "-r", requirements_file], check=True)
            else:
                raise ValueError(f"There is no requirements.txt for {model['name']}.")
        
        except subprocess.CalledProcessError as e:
            print(f"Error {e} occurred during setup for {model['name']}.")
        except Exception as e:
            print(f"Error {e} occurred during setup for {model['name']}.")

if __name__ == "__main__":
    create_folders()
    set_up()
    print("Models were downloaded successfully.")
