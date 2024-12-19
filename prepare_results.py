import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from PIL import Image
import random
import shutil

def create_disparity_heuristics(dir):
    # Define patterns for race and age ANOVA results.
    race_pattern = os.path.join("results/analyze_colorization", "*", "ANOVA_tests", "*", "race.csv")
    age_pattern = os.path.join("results/analyze_colorization", "*", "ANOVA_tests", "*", "age.csv")
    
    # Initialize dictionaries to track counts.
    race_significant_counts = {}
    race_total_metrics = {}
    
    age_significant_counts = {}
    age_total_metrics = {}
    
    def process_files(file_pattern, sig_counts_dict, total_counts_dict):
        for filepath in glob.glob(file_pattern):
            # Extract model and metric from the path.
            parts = filepath.split(os.sep)
            model = parts[4]
            
            # Read the CSV.
            df = pd.read_csv(filepath)
            
            # Increment total metrics count for this model.
            total_counts_dict[model] = total_counts_dict.get(model, 0) + 1
            
            # If empty or no p-value column, skip significance check.
            if df.empty or "p-value" not in df.columns:
                continue
            
            # Check if there's any significant p-value in the file.
            if (df["p-value"] < 0.05).any():
                sig_counts_dict[model] = sig_counts_dict.get(model, 0) + 1
    
    # Process race and age files.
    process_files(race_pattern, race_significant_counts, race_total_metrics)
    process_files(age_pattern, age_significant_counts, age_total_metrics)
    
    # Compute proportions for race.
    race_rows = []
    for model in race_total_metrics.keys():
        total = race_total_metrics[model]
        sig = race_significant_counts.get(model, 0)
        proportion = sig / total if total > 0 else 0.0
        race_rows.append({"Model": model, "Total_Metrics": total, "Significant_Count": sig, "Proportion_Significant": proportion})
    
    # Compute proportions for age.
    age_rows = []
    for model in age_total_metrics.keys():
        total = age_total_metrics[model]
        sig = age_significant_counts.get(model, 0)
        proportion = sig / total if total > 0 else 0.0
        age_rows.append({"Model": model, "Total_Metrics": total, "Significant_Count": sig, "Proportion_Significant": proportion})
    
    # Convert to DataFrames.
    race_df = pd.DataFrame(race_rows)
    age_df = pd.DataFrame(age_rows)
    
    # Ensure directories exist.
    os.makedirs(dir, exist_ok=True)
    
    # Save the results.
    race_heuristic_path = os.path.join(dir, "race_disparity_heuristic.csv")
    age_heuristic_path = os.path.join(dir, "age_disparity_heuristic.csv")
    race_df.to_csv(race_heuristic_path, index=False)
    age_df.to_csv(age_heuristic_path, index=False)

def create_disparity_barplots(base_dir):
    # Define paths to the heuristic CSVs.
    race_path = os.path.join(base_dir, "race_disparity_heuristic.csv")
    age_path = os.path.join(base_dir, "age_disparity_heuristic.csv")
    
    # Read the data.
    if not os.path.exists(race_path) or not os.path.exists(age_path):
        print("Heuristic CSV files not found. Please run the heuristic creation first.")
        return
    
    df_race = pd.read_csv(race_path)
    df_age = pd.read_csv(age_path)
    
    # Extract the year from the Model column.
    df_race['Year'] = df_race['Model'].apply(lambda x: int(x.split()[0]))
    df_age['Year'] = df_age['Model'].apply(lambda x: int(x.split()[0]))
    
    # Sort by year.
    df_race = df_race.sort_values('Year')
    df_age = df_age.sort_values('Year')
    
    # Plot race.
    plt.figure()
    sns.barplot(x="Model", y="Proportion_Significant", data=df_race, order=df_race["Model"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Proportion of Metrics with Significant Race Differences by Model Year")
    plt.tight_layout()
    race_plot_path = os.path.join(base_dir, "race_disparity_barplot.png")
    plt.savefig(race_plot_path)
    plt.close()
    
    # Plot age.
    plt.figure()
    sns.barplot(x="Model", y="Proportion_Significant", data=df_age, order=df_age["Model"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Proportion of Metrics with Significant Age Differences by Model Year")
    plt.tight_layout()
    age_plot_path = os.path.join(base_dir, "age_disparity_barplot.png")
    plt.savefig(age_plot_path)
    plt.close()

def create_recolorization_grid():
    base_colorized_dir = "data/colorized"
    models = ["2001_reinhard", "2016_zhang", "2017_zhang", "2018_antic", "2023_kang"]
    
    # Verify that all models exist.
    for m in models:
        model_path = os.path.join(base_colorized_dir, m)
        if not os.path.isdir(model_path):
            print(f"Model directory not found: {model_path}")
            return
    
    first_model_dir = os.path.join(base_colorized_dir, models[0])
    race_gender_age_dirs = [d for d in os.listdir(first_model_dir) if os.path.isdir(os.path.join(first_model_dir,d))]
    race_gender_age_dirs.sort()
    
    race_gender_map = {}
    for rgad in race_gender_age_dirs:
        parts = rgad.split("_")
        if len(parts) < 3:
            continue
        race = parts[0]
        gender = parts[1]
        age = "_".join(parts[2:])
        rg_key = (race, gender)
        if rg_key not in race_gender_map:
            race_gender_map[rg_key] = []
        race_gender_map[rg_key].append(age)
    
    # Sort the keys before shuffling for stable input to random.
    all_rg = list(race_gender_map.keys())
    all_rg.sort()  
    if len(all_rg) < 14:
        print(f"There should be at least 14 race-gender combos.")
    else:
        random.shuffle(all_rg)
        all_rg = all_rg[:14]
    
    def get_subjects_for_model_rg(mdir, race, gender, age):
        rgad = f"{race}_{gender}_{age}"
        rgad_dir = os.path.join(mdir, rgad)
        if not os.path.isdir(rgad_dir):
            return set()
        fnames = os.listdir(rgad_dir)
        fnames.sort()
        subs = set()
        for fname in fnames:
            if fname.endswith(".jpg"):
                base = os.path.splitext(fname)[0] 
                if "_" in base:
                    dataset, number = base.split("_",1)
                    subs.add((dataset, number))
        return subs
    
    results = []
    for (race, gender) in all_rg:
        ages = race_gender_map[(race, gender)]
        ages.sort()
        random.shuffle(ages)
        chosen_age = None
        chosen_subject = None
        
        for age in ages:
            all_model_subs = []
            for m in models:
                mdir = os.path.join(base_colorized_dir, m)
                subs = get_subjects_for_model_rg(mdir, race, gender, age)
                if not subs:
                    all_model_subs = []
                    break
                all_model_subs.append(subs)
            
            if len(all_model_subs) == len(models):
                common = all_model_subs[0]
                for s in all_model_subs[1:]:
                    common = common.intersection(s)
                if common:
                    common_list = list(common)
                    common_list.sort()
                    chosen_age = age
                    chosen_subject = random.choice(common_list)
                    break
        
        results.append((race, gender, chosen_age, chosen_subject))
    
    fig, axes = plt.subplots(nrows=14, ncols=6, figsize=(18, 42))
    
    for i, (race, gender, age, subject) in enumerate(results):
        if subject is None:
            for ax in axes[i,:]:
                ax.axis("off")
            continue
        
        dataset, number = subject
        gt_path_train = os.path.join("data", "train", f"{number}.jpg")
        gt_path_val = os.path.join("data", "val", f"{number}.jpg")
        
        if os.path.exists(gt_path_train):
            gt_path = gt_path_train
        elif os.path.exists(gt_path_val):
            gt_path = gt_path_val
        else:
            gt_path = None
        
        if gt_path and os.path.exists(gt_path):
            img_gt = Image.open(gt_path).convert("RGB")
        else:
            img_gt = Image.new("RGB", (100,100), color='gray')
        
        axes[i,0].imshow(img_gt)
        axes[i,0].set_title("Ground Truth", fontsize=10)
        axes[i,0].axis("off")
        
        rgad = f"{race}_{gender}_{age}"
        
        for j, m in enumerate(models):
            mdir = os.path.join(base_colorized_dir, m, rgad)
            fname = f"{dataset}_{number}.jpg"
            cpath = os.path.join(mdir, fname)
            if os.path.exists(cpath):
                img_col = Image.open(cpath).convert("RGB")
            else:
                img_col = Image.new("RGB", (100,100), color='red')
            axes[i, j+1].imshow(img_col)
            axes[i, j+1].set_title(m, fontsize=10)
            axes[i, j+1].axis("off")
        
        axes[i,0].set_ylabel(f"{race}-{gender}", fontsize=10)
    
    plt.tight_layout()
    out_dir = "results/prepare_results"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "recolorization_grid.png"))
    plt.close()

# Define main script.
if __name__ == "__main__":
    random.seed(2831 * 2831)
    analysis_dir = "results/prepare_results"

    # Prepare results.
    create_disparity_heuristics(analysis_dir)
    create_disparity_barplots(analysis_dir)
    create_recolorization_grid()
