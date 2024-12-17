import os
import csv
import pyiqa
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_group(folder_name):
    parts = folder_name.split('_')
    gender_code = parts[0]
    race_code = parts[1]
    age_code = parts[-1]

    gender = "Male" if gender_code == 'M' else "Female"

    race_mapping = {
        'B': 'Black',
        'E': 'East Asian',
        'I': 'Indian',
        'LH': 'Latino_Hispanic',
        'M': 'Middle Eastern',
        'S': 'Southeast Asian',
        'W': 'White',
    }
    race = race_mapping.get(race_code, race_code)

    if age_code == '70plus':
        age = "more than 70"
    else:
        age = age_code.replace('to', '-')

    return gender, race, age

def format_model_name(model_name):
    parts = model_name.split('_')
    return ' '.join([part.capitalize() if not part.isdigit() else part for part in parts])

fr_metrics = [
    ("TOPIQ (FR)", "topiq_fr"),
    ("AHIQ", "ahiq"),
    ("PieAPP", "pieapp"),
    ("LPIPS", "lpips"),
    ("DISTS", "dists"),
    ("WaDIQaM (FR)", "wadiqam_fr"),
    ("CKDN1", "ckdn"),
    ("FSIM", "fsim"),
    ("SSIM", "ssimc"),
    ("MS-SSIM", "ms_ssim"),
    ("CW-SSIM", "cw_ssim"),
    ("PSNR", "psnr"),
    ("VIF", "vif"),
    ("GMSD", "gmsd"),
    ("NLPD", "nlpd"),
    ("VSI", "vsi"),
    ("MAD", "mad"),
]

nr_metrics = [
    ("Q-Align", "qalign"),
    ("LIQE", "liqe"),
    ("ARNIQA", "arniqa"),
    ("TOPIQ (NR)", "topiq_nr"),
    ("TReS", "tres"),
    ("FID", "fid"),
    ("CLIPIQA", "clipiqa"),
    ("MANIQA", "maniqa"),
    ("MUSIQ", "musiq"),
    ("DBCNN", "dbcnn"),
    ("PaQ-2-PiQ", "paq2piq"),
    ("HyperIQA", "hyperiqa"),
    ("NIMA", "nima"),
    ("WaDIQaM (NR)", "wadiqam_nr"),
    ("CNNIQA", "cnniqa"),
    ("NRQM", "nrqm"),
    ("PI", "pi"),
    ("BRISQUE", "brisque"),
    ("ILNIQE", "ilniqe"),
    ("NIQE", "niqe"),
    ("PIQE", "piqe"),
]

fr_model_names = {model_name for (_, model_name) in fr_metrics}
nr_model_names = {model_name for (_, model_name) in nr_metrics}

metrics_list = fr_metrics + nr_metrics

colorized_root = 'data/colorized'
results_dir = 'results/compute_metrics'
os.makedirs(results_dir, exist_ok=True)

# There are 5 models and 126 demographic groups with 22 images each.
total_images = 5 * 126 * 22

for method_name, model_name in metrics_list:
    iqa_metric = pyiqa.create_metric(model_name, device=device)

    if model_name in fr_model_names:
        is_fr = True
    elif model_name in nr_model_names:
        is_fr = False
    else:
        raise ValueError(f"Model name {model_name} was not found in FR or NR lists.")

    output_csv = os.path.join(results_dir, f"{method_name}.csv")
    existing_entries = set()

    # Load existing entries.
    if os.path.exists(output_csv):
        with open(output_csv, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and "Dataset" in header and "Number" in header and "Model" in header:
                dataset_idx = header.index("Dataset")
                number_idx = header.index("Number")
                model_idx = header.index("Model")
                for row in reader:
                    if len(row) > model_idx:
                        dataset_val = row[dataset_idx]
                        number_val = row[number_idx]
                        model_val = row[model_idx]
                        existing_entries.add((dataset_val, number_val, model_val))

    # Count how many images have already been processed.
    already_processed_images = len(existing_entries)
    total_images_to_process_now = total_images - already_processed_images

    # Start fresh for this run.
    processed_images = 0
    metric_start_time = time.time()

    if total_images_to_process_now <= 0:
        print(f"{method_name} has already been processed.")
        continue

    write_header = not os.path.exists(output_csv)

    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Dataset", "Number", "Age", "Gender", "Race", "Model", "Score", "Metric", "Lower Better"])

        # Loop through each of the 5 model directories.
        for model_dir_name in os.listdir(colorized_root):
            if model_dir_name.startswith('.'):
                continue
            model_dir = os.path.join(colorized_root, model_dir_name)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Model directory {model_dir} was not found.")

            formatted_model = format_model_name(model_dir_name)

            # Loop through each of the 126 demographic folders.
            for demographic_folder in os.listdir(model_dir):
                if demographic_folder.startswith('.'):
                    continue
                demo_dir = os.path.join(model_dir, demographic_folder)
                if not os.path.isdir(demo_dir):
                    raise FileNotFoundError(f"Demographic directory {demo_dir} was not found.")

                gender, race, age = decode_group(demographic_folder)

                # Process each image in the demographic folder.
                for img_name in os.listdir(demo_dir):
                    if img_name.startswith('.'):
                        continue
                    if not img_name.lower().endswith(('.jpg', '.png', '.bmp')):
                        raise ValueError(f"Image {img_name} is invalid.")

                    distorted_img_path = os.path.join(demo_dir, img_name)
                    prefix, number_ext = img_name.split('_', 1)
                    dataset = prefix
                    number_str = os.path.splitext(number_ext)[0]

                    key = (dataset, number_str, formatted_model)
                    if key in existing_entries:
                        # If it was already processed in a previous run, skip.
                        continue

                    # Compute IQA score for this image.
                    if is_fr:
                        ref_img_path = os.path.join('data', dataset, f"{number_str}.jpg")
                        if not os.path.exists(ref_img_path):
                            raise FileNotFoundError(f"Reference image {ref_img_path} was not found.")
                        score = iqa_metric(distorted_img_path, ref_img_path)
                    else:
                        score = iqa_metric(distorted_img_path)

                    if torch.is_tensor(score):
                        score = score.item()

                    # Write row.
                    writer.writerow([
                        dataset,
                        number_str,
                        age,
                        gender,
                        race,
                        formatted_model,
                        score,
                        method_name,
                        iqa_metric.lower_better
                    ])
                    existing_entries.add(key)

                    # Increment count for this session.
                    processed_images += 1

                    # Estimate remaining time based on this session.
                    elapsed_time = time.time() - metric_start_time
                    avg_time_per_image = elapsed_time / processed_images
                    remaining_images = total_images_to_process_now - processed_images
                    estimated_remaining_time = avg_time_per_image * remaining_images / 60

                    print(f"{method_name} has processed {already_processed_images + processed_images}/{total_images} images with {estimated_remaining_time:.2f} minutes remaining.")
