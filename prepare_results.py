import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

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

# Define main script.
if __name__ == "__main__":
    analysis_dir = "results/prepare_results"

    # Prepare results.
    create_disparity_heuristics(analysis_dir)
    create_disparity_barplots(analysis_dir)
