import os
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import random

PROGRESS_LOG = "progress.log"

def load_completed_scripts():
    if os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG, "r") as file:
            return set(file.read().splitlines())
    return set()

def save_completed_script(name):
    with open(PROGRESS_LOG, "a") as file:
        file.write(name + "\n")

def run_scripts(directory):
    scripts = sorted([f for f in os.listdir(directory) if f.endswith(".py")])

    completed_scripts = load_completed_scripts()

    for script in scripts:
        if script not in completed_scripts:
            script_path = os.path.join(directory, script)
            print(f"Script {script_path} is being run.")

            try:
                subprocess.run(["python", script_path], check=True)
                print(f"Script {script_path} is finished running.")
                save_completed_script(script)
            except subprocess.CalledProcessError as e:
                print(f"Error {e} occurred while running script {script_path}.")

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
    out_dir = "results/run_colorization"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "recolorization_grid.png"))
    plt.close()

if __name__ == "__main__":
    run_scripts("scripts/")
    random.seed(2831 * 2831)
    create_recolorization_grid()
