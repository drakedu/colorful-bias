import os
import pandas as pd
import matplotlib.pyplot as plt
import joypy
from itertools import chain, combinations
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, shapiro, t, mannwhitneyu, f_oneway, levene

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

def run_t_tests_for_metric(df_metric, metric, analysis_dir):
    metric_dir = os.path.join(analysis_dir, metric)
    os.makedirs(metric_dir, exist_ok=True)
    t_test_dir = os.path.join(metric_dir, "t_tests")
    os.makedirs(t_test_dir, exist_ok=True)

    models = df_metric["Model"].unique()

    def check_normality(x):
        x = np.array(x)
        if len(x) < 3:
            return np.nan, False
        w_stat, p_val = shapiro(x)
        normal = (p_val > 0.05)
        return p_val, normal

    def welch_t_confidence_interval(mean_diff, var1, var2, n1, n2, alpha=0.05):
        se_diff = np.sqrt(var1/n1 + var2/n2)
        # Handle degenerate case.
        if se_diff == 0:
            return mean_diff, mean_diff, se_diff
        # COmpute Welch-Satterthwaite degrees of freedom.
        df = ((var1/n1 + var2/n2)**2) / ((var1**2 / (n1**2*(n1-1))) + (var2**2 / (n2**2*(n2-1))))
        t_crit = t.ppf(1 - alpha/2, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        return ci_lower, ci_upper, se_diff

    def do_comparison(g1_scores, g2_scores, comparison_name, group1_name, group2_name):
        n1 = len(g1_scores)
        n2 = len(g2_scores)
        if n1 < 2 or n2 < 2:
            # We have insuficient data.
            return None
        
        # Check normality.
        shapiro_p1, normal1 = check_normality(g1_scores)
        shapiro_p2, normal2 = check_normality(g2_scores)

        # Conduct Welch t-test.
        t_stat, p_val = ttest_ind(g1_scores, g2_scores, equal_var=False)
        
        mean1 = np.mean(g1_scores)
        mean2 = np.mean(g2_scores)
        mean_diff = mean1 - mean2
        var1 = np.var(g1_scores, ddof=1)
        var2 = np.var(g2_scores, ddof=1)

        ci_lower, ci_upper, se_diff = welch_t_confidence_interval(mean_diff, var1, var2, n1, n2)

        return {
            "Comparison": comparison_name,
            "Assumptions": normal1 and normal2,
            "Group1": group1_name,
            "Group2": group2_name,
            "N1": n1,
            "N2": n2,
            "Mean1": mean1,
            "Mean2": mean2,
            "Mean_Diff": mean_diff,
            "SE_Mean_Diff": se_diff,
            "t-stat": t_stat,
            "p-value": p_val,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
            "Shapiro_pval_Group1": shapiro_p1,
            "Group1_Normal": normal1,
            "Shapiro_pval_Group2": shapiro_p2,
            "Group2_Normal": normal2,
        }

    for model in models:
        model_df = df_metric[df_metric["Model"] == model]
        if model_df.empty:
            continue

        out_path = os.path.join(t_test_dir, f"{model}.csv")

        # Skip if file already exists.
        if os.path.exists(out_path):
            continue

        results = []

        # 1. Check male versus female.
        male_scores = model_df[model_df["Gender"] == "Male"]["Score"]
        female_scores = model_df[model_df["Gender"] == "Female"]["Score"]
        if not male_scores.empty and not female_scores.empty:
            res = do_comparison(male_scores, female_scores, "Male vs Female", "Male", "Female")
            if res is not None:
                results.append(res)

        # 2. Check each age versus the test.
        unique_ages = model_df["Age"].unique()
        for age_val in unique_ages:
            age_scores = model_df[model_df["Age"] == age_val]["Score"]
            rest_scores = model_df[model_df["Age"] != age_val]["Score"]
            if not age_scores.empty and not rest_scores.empty:
                comp_name = f"{age_val} vs Rest"
                res = do_comparison(age_scores, rest_scores, comp_name, age_val, "Rest")
                if res is not None:
                    results.append(res)

        # 3. Check white against each other race/ethnicity.
        if "White" in model_df["Race"].values:
            white_scores = model_df[model_df["Race"] == "White"]["Score"]
            other_races = [r for r in model_df["Race"].unique() if r != "White"]
            for orace in other_races:
                orace_scores = model_df[model_df["Race"] == orace]["Score"]
                if not white_scores.empty and not orace_scores.empty:
                    comp_name = f"White vs {orace}"
                    res = do_comparison(white_scores, orace_scores, comp_name, "White", orace)
                    if res is not None:
                        results.append(res)

            # 4. Check white versus non-white.
            nonwhite_scores = model_df[model_df["Race"] != "White"]["Score"]
            if not white_scores.empty and not nonwhite_scores.empty:
                comp_name = "White vs Non-White"
                res = do_comparison(white_scores, nonwhite_scores, comp_name, "White", "Non-White")
                if res is not None:
                    results.append(res)

        # Save results to CSV.
        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)

def run_mannwhitney_for_metric(df_metric, metric, analysis_dir):
    metric_dir = os.path.join(analysis_dir, metric)
    os.makedirs(metric_dir, exist_ok=True)
    t_test_dir = os.path.join(metric_dir, "mannwhitneyu_tests")
    os.makedirs(t_test_dir, exist_ok=True)

    models = df_metric["Model"].unique()

    def rank_biserial_correlation(U, n1, n2):
        return 1 - (2 * U) / (n1 * n2)

    def do_comparison(g1_scores, g2_scores, comparison_name, group1_name, group2_name):
        n1 = len(g1_scores)
        n2 = len(g2_scores)
        if n1 < 2 or n2 < 2:
            return None
        
        # Use two-sided Mann-Whitney U test.
        U_stat, p_val = mannwhitneyu(g1_scores, g2_scores, alternative='two-sided')
        
        median1 = np.median(g1_scores)
        median2 = np.median(g2_scores)
        r = rank_biserial_correlation(U_stat, n1, n2)

        return {
            "Comparison": comparison_name,
            "Group1": group1_name,
            "Group2": group2_name,
            "N1": n1,
            "N2": n2,
            "Median1": median1,
            "Median2": median2,
            "U-stat": U_stat,
            "p-value": p_val,
            "RankBiserialCorr": r
        }

    for model in models:
        model_df = df_metric[df_metric["Model"] == model]
        if model_df.empty:
            continue

        out_path = os.path.join(t_test_dir, f"{model}.csv")

        # Skip if file already exists.
        if os.path.exists(out_path):
            continue

        results = []

        # 1. Check male versus female.
        male_scores = model_df[model_df["Gender"] == "Male"]["Score"]
        female_scores = model_df[model_df["Gender"] == "Female"]["Score"]
        if not male_scores.empty and not female_scores.empty:
            res = do_comparison(male_scores, female_scores, "Male vs Female", "Male", "Female")
            if res is not None:
                results.append(res)

        # 2. Check each age versus the rest.
        unique_ages = model_df["Age"].unique()
        for age_val in unique_ages:
            age_scores = model_df[model_df["Age"] == age_val]["Score"]
            rest_scores = model_df[model_df["Age"] != age_val]["Score"]
            if not age_scores.empty and not rest_scores.empty:
                comp_name = f"{age_val} vs Rest"
                res = do_comparison(age_scores, rest_scores, comp_name, age_val, "Rest")
                if res is not None:
                    results.append(res)

        # 3. Check white versus each other race/ethnicity.
        if "White" in model_df["Race"].values:
            white_scores = model_df[model_df["Race"] == "White"]["Score"]
            other_races = [r for r in model_df["Race"].unique() if r != "White"]
            for orace in other_races:
                orace_scores = model_df[model_df["Race"] == orace]["Score"]
                if not white_scores.empty and not orace_scores.empty:
                    comp_name = f"White vs {orace}"
                    res = do_comparison(white_scores, orace_scores, comp_name, "White", orace)
                    if res is not None:
                        results.append(res)

            # 4. Check white versus non-white.
            nonwhite_scores = model_df[model_df["Race"] != "White"]["Score"]
            if not white_scores.empty and not nonwhite_scores.empty:
                comp_name = "White vs Non-White"
                res = do_comparison(white_scores, nonwhite_scores, comp_name, "White", "Non-White")
                if res is not None:
                    results.append(res)

        # Save results to CSV.
        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)

def run_anovas_for_metric(df_metric, metric, analysis_dir):
    metric_dir = os.path.join(analysis_dir, metric)
    anova_dir = os.path.join(metric_dir, "ANOVA")
    os.makedirs(anova_dir, exist_ok=True)

    def group_stats_and_checks(groups):
        stats = {}
        for gname, scores in groups.items():
            arr = np.array(scores)
            n = len(arr)
            mean = np.mean(arr) if n > 0 else np.nan
            std = np.std(arr, ddof=1) if n > 1 else np.nan
            if n >= 3:
                w_stat, p_val = shapiro(arr)
                normal = p_val > 0.05
            else:
                p_val = np.nan
                normal = False
            stats[gname] = {
                "N": n,
                "Mean": mean,
                "Std": std,
                "Shapiro_p": p_val,
                "Normal": normal
            }
        return stats

    def perform_anova(groups, factor):
        if len(groups) < 2:
            return None
        
        # Filter out groups with fewer than 2 observations.
        filtered_groups = {g: arr for g, arr in groups.items() if len(arr) >= 2}
        if len(filtered_groups) < 2:
            return None

        stats = group_stats_and_checks(filtered_groups)
        if len(stats) < 2:
            return None

        # Perform Levene’s test.
        group_arrays = list(filtered_groups.values())
        if all(len(g) > 1 for g in group_arrays):
            lev_stat, lev_p = levene(*group_arrays)
        else:
            lev_p = np.nan

        # Perform ANOVA.
        f_stat, p_val = f_oneway(*group_arrays)

        all_scores = np.concatenate(group_arrays)
        total_n = len(all_scores)
        k = len(filtered_groups)
        df_between = k - 1
        df_within = total_n - k

        row = {
            "Factor": factor,
            "Groups": ",".join(filtered_groups.keys()),
            "df_between": df_between,
            "df_within": df_within,
            "F": f_stat,
            "p-value": p_val,
            "Levene_p": lev_p
        }
        for i, (gname, st) in enumerate(stats.items(), start=1):
            row[f"GroupName_{i}"] = gname
            row[f"N_{i}"] = st["N"]
            row[f"Mean_{i}"] = st["Mean"]
            row[f"Std_{i}"] = st["Std"]
            row[f"Shapiro_p_{i}"] = st["Shapiro_p"]
            row[f"Normal_{i}"] = st["Normal"]
        return row

    # 1. Run ANOVA overall by model.
    overall_path = os.path.join(anova_dir, "overall.csv")
    if not os.path.exists(overall_path):
        model_groups = {}
        for model_val in df_metric["Model"].unique():
            model_groups[model_val] = df_metric[df_metric["Model"] == model_val]["Score"].values
        overall_res = perform_anova(model_groups, factor="Model")
        if overall_res is not None:
            pd.DataFrame([overall_res]).to_csv(overall_path, index=False)

    # 2. Run ANOVA for each model by race and age.
    for model_val in df_metric["Model"].unique():
        model_subset = df_metric[df_metric["Model"] == model_val]
        model_anova_dir = os.path.join(anova_dir, model_val)
        os.makedirs(model_anova_dir, exist_ok=True)

        race_path = os.path.join(model_anova_dir, "race.csv")
        if not os.path.exists(race_path):
            race_groups = {}
            for r_val in model_subset["Race"].unique():
                race_groups[r_val] = model_subset[model_subset["Race"] == r_val]["Score"].values
            race_res = perform_anova(race_groups, factor="Race")
            if race_res is not None:
                pd.DataFrame([race_res]).to_csv(race_path, index=False)

        age_path = os.path.join(model_anova_dir, "age.csv")
        if not os.path.exists(age_path):
            age_groups = {}
            for a_val in model_subset["Age"].unique():
                age_groups[a_val] = model_subset[model_subset["Age"] == a_val]["Score"].values
            age_res = perform_anova(age_groups, factor="Age")
            if age_res is not None:
                pd.DataFrame([age_res]).to_csv(age_path, index=False)

    # 3. Run ANOVA by model for each subsets of data: each non-white race/ethnicity and all non-white images combined.
    nonwhite_path = os.path.join(anova_dir, "race.csv")
    # If file exists, skip.
    if not os.path.exists(nonwhite_path):
        all_races = df_metric["Race"].unique()
        nonwhite_races = [r for r in all_races if r.lower() != "white"]
        results_nonwhite = []

        # Get nonwhite combined.
        nonwhite_subset = df_metric[~df_metric["Race"].str.lower().str.contains("white")]
        if len(nonwhite_subset) > 0:
            nonwhite_groups = {}
            for model_val in nonwhite_subset["Model"].unique():
                nonwhite_groups[model_val] = nonwhite_subset[nonwhite_subset["Model"] == model_val]["Score"].values
            res_nonwhite = perform_anova(nonwhite_groups, factor="Model")
            if res_nonwhite is not None:
                # Indicate which subset we are testing.
                res_nonwhite["Subset"] = "Non-White"
                results_nonwhite.append(res_nonwhite)

        # Loop over each nonwhite race.
        for nr in nonwhite_races:
            nr_subset = df_metric[df_metric["Race"] == nr]
            nr_groups = {}
            for model_val in nr_subset["Model"].unique():
                nr_groups[model_val] = nr_subset[nr_subset["Model"] == model_val]["Score"].values
            res_nr = perform_anova(nr_groups, factor="Model")
            if res_nr is not None:
                # Indicate which subset or particular nonwhite race.
                res_nr["Subset"] = nr
                results_nonwhite.append(res_nr)

        # Save all nonwhite results into one CSV.
        if results_nonwhite:
            pd.DataFrame(results_nonwhite).to_csv(nonwhite_path, index=False)

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
        run_t_tests_for_metric(df, metric, analysis_dir)
        run_mannwhitney_for_metric(df, metric, analysis_dir)
        run_anovas_for_metric(df, metric, analysis_dir)
