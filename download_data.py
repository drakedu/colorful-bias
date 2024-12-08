import gdown
import re
import os
import zipfile
import pandas as pd

# Define URLs.
train_label = "https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view"
val_label = "https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view"
train_val_img = "https://drive.google.com/file/d/1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL/view"

# Create "data" directory if it doesn't exist.
def create_data_folder():
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    return data_folder

# Get file ID from Google Drive URL.
def extract_file_id(drive_url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Google Drive URL {drive_url} is invalid.")

# Download file from Google Drive.
def download_from_drive(public_url, output_path):
    try:
        file_id = extract_file_id(public_url)
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, output_path, quiet=False)
    except Exception as e:
        print(f"Error {e} occurred while downloading file from Google Drive with URL {public_url}.")

# Unzip the downloaded file and delete the zip file.
def unzip_and_cleanup(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
    except Exception as e:
        print(f"Error {e} occurred while unzipping file {zip_path}.")

# Create label CSV file that merges training and validation labels.
def combine_labels(train_label_path, val_label_path):
    train_label = pd.read_csv(train_label_path)
    val_label = pd.read_csv(val_label_path)
    combined_label = pd.concat([train_label, val_label], ignore_index=True)
    combined_label.to_csv("data/combined_label.csv", index=False)

# Get data.
if __name__ == "__main__":
    data_folder = create_data_folder()
    train_label_path = os.path.join(data_folder, "train_label.csv")
    val_label_path = os.path.join(data_folder, "val_label.csv")
    train_val_img_path = os.path.join(data_folder, "train_val_img.zip")
    download_from_drive(train_label, train_label_path)
    download_from_drive(val_label, val_label_path)
    download_from_drive(train_val_img, train_val_img_path)
    unzip_and_cleanup(train_val_img_path, data_folder)
    combine_labels(train_label_path, val_label_path)
    print("Data was successfully downloaded and unzipped.")
