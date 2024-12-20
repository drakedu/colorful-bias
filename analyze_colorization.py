import os
import pandas as pd
import matplotlib.pyplot as plt
import joypy
from itertools import chain, combinations
import seaborn as sns
import arviz as az
import pymc as pm
import numpy as np
from patsy import dmatrix

def run_multivariate_mixed_effects_model(metric_dir, analysis_dir):
    data_csv_path = os.path.join(analysis_dir, "multivariate_mixed_effects_model_data.csv")

    # Check if processed data file already exists before doing any processing.
    if os.path.exists(data_csv_path):
        # If it exists, just load it.
        df_wide = pd.read_csv(data_csv_path)
    else:
        # Step 1: Combine CSVs only if we need to create the data file.
        csv_files = [f for f in os.listdir(metric_dir) if f.endswith('.csv')]
        df_all = pd.concat([pd.read_csv(os.path.join(metric_dir, f)) for f in csv_files], ignore_index=True)

        if 'Metric' not in df_all.columns or 'Score' not in df_all.columns:
            raise ValueError("The combined DataFrame must contain 'Metric' and 'Score' columns.")
        if 'Dataset' not in df_all.columns or 'Number' not in df_all.columns:
            raise ValueError("The combined DataFrame must contain 'Dataset' and 'Number' columns.")

        # Step 2: Pivot to wide format.
        df_wide = df_all.pivot_table(
            index=['Dataset', 'Number', 'Age', 'Gender', 'Race', 'Model'],
            columns='Metric',
            values='Score'
        ).reset_index()

        # Save the processed data before model fitting.
        df_wide.to_csv(data_csv_path, index=False)

    key_cols = ['Dataset', 'Number', 'Age', 'Gender', 'Race', 'Model']
    metrics = [col for col in df_wide.columns if col not in key_cols]

    for col in ['Dataset', 'Number', 'Age', 'Gender', 'Race', 'Model']:
        df_wide[col] = df_wide[col].astype('category')

    Y = df_wide[metrics].values
    M, num_metrics = Y.shape

    # Add subject indexing.
    df_wide["Subject"] = df_wide["Dataset"].astype(str) + "_" + df_wide["Number"].astype(str)
    df_wide["Subject"] = df_wide["Subject"].astype('category')
    subject_idx = df_wide["Subject"].cat.codes.values
    num_subjects = df_wide["Subject"].nunique()

    # Step 3: Design matrix with interactions.
    formula = "Race * Model"
    X = dmatrix("0 + " + formula, data=df_wide, return_type='dataframe')
    X = pd.concat([pd.DataFrame({"Intercept": np.ones(M)}), X], axis=1)
    X_mat = X.values
    M, P = X_mat.shape

    # Step 4: Fit the Bayesian multivariate mixed model in PyMC.
    with pm.Model() as model:
        mu_subject = pm.Normal("mu_subject", mu=0, sigma=1, shape=(num_subjects, num_metrics))
        beta = pm.Normal("beta", mu=0, sigma=1, shape=(P, num_metrics))
        mu = pm.Deterministic("mu", X_mat @ beta + mu_subject[subject_idx, :])

        # Use compute_corr=True to get chol directly.
        chol, corr, sigmas = pm.LKJCholeskyCov(
            "chol_cov",
            n=num_metrics,
            eta=2.0,
            sd_dist=pm.Exponential.dist(1.0),
            compute_corr=True
        )

        y_obs = pm.MvNormal("y_obs", mu=mu, chol=chol, observed=Y)

        trace = pm.sample(draws=1000, chains=2, target_accept=0.9)

    summary_df = az.summary(trace)
    summary_path = os.path.join(analysis_dir, "multivariate_mixed_effects_model_results.csv")
    summary_df.to_csv(summary_path)

# Encode a row into a plot label based on the subset of columns included.
def encode_attribute(row, subset):
    parts = []
    if "Gender" in subset:
        gender_code = "M" if row["Gender"] == "Male" else "F"
        parts.append(gender_code)
    if "Race" in subset:
        race_code = ''.join([word[0].upper() for word in row["Race"].split("_")])
        parts.append(race_code)
    if "Age" in subset:
        age = row["Age"]
        if age == "more than 70":
            age_code = "70plus"
        else:
            age_code = age.replace("-", "to")
        parts.append(age_code)
    if "Model" in subset:
        model_code = row["Model"].replace(" ", "")
        parts.append(model_code)
    return "_".join(parts) if parts else "All"

# Generate all subsets of the given iterable, including empty set.
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# Generate a filename-friendly name for a given subset of columns.
def subset_name(subset):
    if len(subset) == 0:
        return "None"
    return "_".join([col.lower() for col in subset])

# Create joyplots for a given metric, grouping by subsets.
def create_joyplots_for_metric(df_metric, metric, analysis_dir):
    metric_dir_path = os.path.join(analysis_dir, metric)
    joyplot_dir = os.path.join(metric_dir_path, "joyplots")
    os.makedirs(joyplot_dir, exist_ok=True)
    
    grouping_vars = ["Age", "Gender", "Race", "Model"]
    
    # Generate all subsets of grouping variables.
    for subset in powerset(grouping_vars):
        subset = list(subset)
        filename = subset_name(subset) + ".png"

        # Skip if the image already exists.
        filepath = os.path.join(joyplot_dir, filename)
        if os.path.exists(filepath):
            continue
        
        # Handle no grouping, which is just one histogram of all scores.
        if len(subset) == 0:
            fig, ax = joypy.joyplot(df_metric, column='Score', by=None, overlap=0, linewidth=1, legend=False)
        else:
            # With grouping, dynamically set figure size based on the number of groups.
            df_metric["GroupLabel"] = df_metric.apply(lambda x: encode_attribute(x, subset), axis=1)
            num_groups = df_metric["GroupLabel"].nunique()
            
            # Determine figure size dynamically. Each group gets additional vertical space.
            height_per_subplot = 0.5
            base_width = 10
            base_height = 1
            fig_height = base_height + num_groups * height_per_subplot
            
            fig, ax = joypy.joyplot(df_metric, by="GroupLabel", column='Score', overlap=0, linewidth=1, legend=False, figsize=(base_width, fig_height))
        
        # Save the plot and apply tight layout.
        plt.savefig(os.path.join(joyplot_dir, filename))
        plt.tight_layout()
        plt.close()

# Create summary statistics where 'All' entries represent no grouping on that attribute.
def create_summary_statistics(df_metric, metric, analysis_dir):
    metric_dir_path = os.path.join(analysis_dir, metric)
    os.makedirs(metric_dir_path, exist_ok=True)
    summary_csv_path = os.path.join(metric_dir_path, "summary_statistics.csv")
    
    # Skip if the file already exists.
    if os.path.exists(summary_csv_path):
        return
    
    grouping_vars = ["Age", "Gender", "Race", "Model"]
    score_col = "Score"
    lower_better_col = "Lower Better"
    
    summary_rows = []
    
    # Assume 'Lower Better' is constant for the metric.
    lower_better_value = df_metric[lower_better_col].iloc[0] if lower_better_col in df_metric.columns else False

    # Generate all subsets of grouping variables.
    for subset in powerset(grouping_vars):
        subset = list(subset)

        # Copy and adjust grouping columns. If not in subset, set to "All".
        df_group = df_metric.copy()
        for var in grouping_vars:
            if var not in subset:
                df_group[var] = "All"
                
        grouped = df_group.groupby(["Age", "Gender", "Race", "Model"])
        
        # Compute summary statistics for each group.
        for (age_val, gender_val, race_val, model_val), group_data in grouped:
            scores = group_data[score_col]
            mean_val = scores.mean()
            median_val = scores.median()
            std_val = scores.std()
            q1 = scores.quantile(0.25)
            q3 = scores.quantile(0.75)
            iqr_val = q3 - q1
            
            summary_rows.append([
                age_val, gender_val, race_val, model_val,
                mean_val, median_val, std_val, iqr_val,
                lower_better_value
            ])
    
    # Create DataFrame and save.
    summary_df = pd.DataFrame(summary_rows, columns=[
        "Age", "Gender", "Race", "Model",
        "Mean", "Median", "Standard Deviation", "IQR", "Lower Better"
    ])
    summary_df.to_csv(summary_csv_path, index=False)

def create_facets(df_metric, metric, analysis_dir):
    metric_dir_path = os.path.join(analysis_dir, metric)
    facet_dir = os.path.join(metric_dir_path, "facets")
    os.makedirs(facet_dir, exist_ok=True)
    
    facet_pairs = [
        ("Race", "Model"),
        ("Gender", "Model"),
        ("Age", "Model"),
        ("Race", "Gender"),
        ("Age", "Race"),
        ("Age", "Gender")
    ]
    
    for var1, var2 in facet_pairs:
        pair_file_name = f"{var1.lower()}_{var2.lower()}.png"
        out_file = os.path.join(facet_dir, pair_file_name)
        
        # If image already exists, skip.
        if os.path.exists(out_file):
            continue
        
        # Create the FacetGrid.
        g = sns.FacetGrid(df_metric, row=var1, col=var2, sharex=True, sharey=True, margin_titles=True)
        
        # Map a KDE plot to each facet.
        g.map(sns.kdeplot, "Score", fill=True)
        
        # Adjust layout to fit titles.
        plt.tight_layout()
        
        # Save and close.
        plt.savefig(out_file)
        plt.close()

def create_barcharts(df_metric, metric, analysis_dir):
    metric_dir_path = os.path.join(analysis_dir, metric)
    barchart_dir = os.path.join(metric_dir_path, "barcharts")
    os.makedirs(barchart_dir, exist_ok=True)
    
    attributes = ["Age", "Gender", "Race", "Model"]
    
    for attribute in attributes:
        out_file = os.path.join(barchart_dir, f"{attribute}.png")
        
        # Skip if the file already exists.
        if os.path.exists(out_file):
            continue
        
        # If the attribute column is empty or doesn't exist, skip.
        if attribute not in df_metric.columns:
            continue
        
        # If no data exists for this attribute, skip.
        if df_metric[attribute].empty:
            continue
        
        # Create a simple bar chart
        plt.figure()
        sns.barplot(data=df_metric, x=attribute, y="Score", errorbar=('ci',95))
        
        # Add a title.
        plt.title(f"{metric} Scores by {attribute}")
        
        # Rotate x-tick labels.
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the figure.
        plt.savefig(out_file)
        plt.close()

# Define main script.
if __name__ == "__main__":
    metric_dir = "results/compute_metrics"
    analysis_dir = "results/analyze_colorization"

    os.makedirs(analysis_dir, exist_ok=True)

    # List all CSV files in metric_dir.
    csv_files = [f for f in os.listdir(metric_dir) if f.endswith('.csv')]

    for filename in csv_files:
        # Extract metric from filename by removing the file extension.
        metric = os.path.splitext(filename)[0]
        
        csv_path = os.path.join(metric_dir, filename)
        df = pd.read_csv(csv_path)

        # Run analysis.
        create_joyplots_for_metric(df, metric, analysis_dir)
        create_summary_statistics(df, metric, analysis_dir)
        create_facets(df, metric, analysis_dir)
        create_barcharts(df, metric, analysis_dir)
    
    # Run multivariate mixed effects model.
    run_multivariate_mixed_effects_model(metric_dir, analysis_dir)
