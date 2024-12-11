import numpy as np
import pandas as pd
import os
from PIL import Image

# The smallest group has 22 images.
MAX_SAMPLES = 22

# Encode gender, race, and age in grayscale subfolder name.
def encode_group(gender, race, age):
    gender_code = "M" if gender == "Male" else "F"
    race_code = ''.join([word[0].upper() for word in race.split("_")])
    if age == "more than 70":
        age_code = "70plus"
    else:
        age_code = age.replace("-", "to")
    return f"{gender_code}_{race_code}_{age_code}"

# Grayscale N random images from each gender-race-age group.
def sample_data(N):
    combined_label_path = "data/combined_label.csv"
    combined_df = pd.read_csv(combined_label_path)
    output_dir = "data/grayscale"
    os.makedirs(output_dir, exist_ok=True)
    grouped = combined_df.groupby(["gender", "race", "age"])
    for (gender, race, age), group in grouped:
        if len(group) < N:
            return ValueError(f"Number of samples {N} exceeds the number of images labeled {gender}, {race}, and {age}.")
        sampled = group.sample(N)
        group_folder = encode_group(gender, race, age)
        group_path = os.path.join(output_dir, group_folder)
        os.makedirs(group_path, exist_ok=True)
        for _, row in sampled.iterrows():
            file_path = os.path.join("data", row["file"])
            prefix, id = row["file"].split("/")[0], row["file"].split("/")[1]
            new_name = f"{prefix}_{id}"
            output_path = os.path.join(group_path, new_name)
            try:
                img = Image.open(file_path).convert("L")
                img.save(output_path)
            except Exception as e:
                print(f"Error {e} occured while processing image {file_path}.")

if __name__ == "__main__":
    # Set seed.
    np.random.seed(2831)

    # Sample data.
    sample_data(MAX_SAMPLES)
