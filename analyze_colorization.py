import os
import pandas as pd
import matplotlib.pyplot as plt
import joypy
from itertools import chain, combinations
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, shapiro, t, mannwhitneyu, fligner
import pingouin as pg
import glob

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
            "GroupName_1": group1_name,
            "GroupName_2": group2_name,
            "N_1": n1,
            "N_2": n2,
            "Mean_1": mean1,
            "Mean_2": mean2,
            "Mean_Diff": mean_diff,
            "SE_Mean_Diff": se_diff,
            "t-stat": t_stat,
            "p-value": p_val,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
            "Shapiro_p_1": shapiro_p1,
            "Normal_1": normal1,
            "Shapiro_p_2": shapiro_p2,
            "Normal_2": normal2,
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
    mannwhitney_dir = os.path.join(metric_dir, "mannwhitneyu_tests")
    os.makedirs(mannwhitney_dir, exist_ok=True)

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
            "GroupName_1": group1_name,
            "GroupName_2": group2_name,
            "N_1": n1,
            "N_2": n2,
            "Median_1": median1,
            "Median_2": median2,
            "U-stat": U_stat,
            "p-value": p_val,
            "RankBiserialCorr": r
        }

    for model in models:
        model_df = df_metric[df_metric["Model"] == model]
        if model_df.empty:
            continue

        out_path = os.path.join(mannwhitney_dir, f"{model}.csv")

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
    # Create SubjectID for repeated-measures and mixed-design analysis.
    df_metric["SubjectID"] = df_metric["Dataset"].astype(str) + "_" + df_metric["Number"].astype(str)

    metric_dir = os.path.join(analysis_dir, metric)
    anova_dir = os.path.join(metric_dir, "ANOVA_tests")
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

    # Define one-way Welch's ANOVA.
    def perform_anova(groups, factor):
        if len(groups) < 2:
            return None

        filtered_groups = {g: arr for g, arr in groups.items() if len(arr) >= 2}
        if len(filtered_groups) < 2:
            return None

        stats = group_stats_and_checks(filtered_groups)
        if len(stats) < 2:
            return None

        # Prepare data for welch_anova.
        data_list = []
        for gname, arr in filtered_groups.items():
            for val in arr:
                data_list.append((gname, val))
        df_long = pd.DataFrame(data_list, columns=["Group", "Score"])

        wres = pg.welch_anova(dv="Score", between="Group", data=df_long)
        if wres.empty:
            return None

        ddof1 = wres.loc[0, "ddof1"]
        ddof2 = wres.loc[0, "ddof2"]
        F_val = wres.loc[0, "F"]
        p_val = wres.loc[0, "p-unc"]

        row = {
            "Factor": factor,
            "Groups": ",".join(filtered_groups.keys()),
            "Test": "Welch_ANOVA",
            "df_between": ddof1,
            "df_within": ddof2,
            "F": F_val,
            "p-value": p_val,
        }
        for i, (gname, st) in enumerate(stats.items(), start=1):
            row[f"GroupName_{i}"] = gname
            row[f"N_{i}"] = st["N"]
            row[f"Mean_{i}"] = st["Mean"]
            row[f"Std_{i}"] = st["Std"]
            row[f"Shapiro_p_{i}"] = st["Shapiro_p"]
            row[f"Normal_{i}"] = st["Normal"]
        return row

    # Define repeated-measures ANOVA.
    def perform_rm_anova(df_long, factor):
        models = df_long["Model"].unique()
        if len(models) < 2:
            return None

        subj_counts = df_long.groupby(["SubjectID", "Model"]).size().unstack(fill_value=0)
        complete_subjects = subj_counts.index[subj_counts.notnull().all(axis=1)]
        df_long = df_long[df_long["SubjectID"].isin(complete_subjects)]

        if len(df_long["SubjectID"].unique()) < 2:
            return None

        groups = {m: df_long[df_long["Model"] == m]["Score"].dropna().values for m in models}
        stats = group_stats_and_checks(groups)
        if len(stats) < 2:
            return None

        res = pg.rm_anova(data=df_long, dv="Score", correction='none', within="Model", subject="SubjectID", detailed=True)
        if res.empty:
            return None

        model_row = res[res["Source"] == "Model"]
        if model_row.empty:
            return None

        p_val = model_row["p-unc"].values[0]
        F_val = model_row["F"].values[0]
        ddof1 = model_row["DF"].values[0]

        row = {
            "Factor": factor,
            "Groups": ",".join(models),
            "Test": "RM_ANOVA",
            "ddof1": ddof1,
            "F": F_val,
            "p-value": p_val,
        }

        for i, (gname, st) in enumerate(stats.items(), start=1):
            row[f"GroupName_{i}"] = gname
            row[f"N_{i}"] = st["N"]
            row[f"Mean_{i}"] = st["Mean"]
            row[f"Std_{i}"] = st["Std"]
            row[f"Shapiro_p_{i}"] = st["Shapiro_p"]
            row[f"Normal_{i}"] = st["Normal"]
        return row

    # Define mixed-design ANOVA.
    def perform_mixed_anova(df_metric, factor, between_factor, filename):
        models = df_metric["Model"].unique()
        if len(models) < 2:
            return None

        between_groups = df_metric[between_factor].unique()
        if len(between_groups) < 2:
            return None

        subj_counts = df_metric.groupby(["SubjectID", "Model"]).size().unstack(fill_value=0)
        complete_subjects = subj_counts.index[subj_counts.notnull().all(axis=1)]
        df_clean = df_metric[df_metric["SubjectID"].isin(complete_subjects)]

        if len(df_clean["SubjectID"].unique()) < 2:
            return None

        # Check normality for each combination of Model x between_group.
        factor_groups = {}
        for m in models:
            for bg in between_groups:
                arr = df_clean[(df_clean["Model"] == m) & (df_clean[between_factor] == bg)]["Score"].dropna().values
                factor_groups[f"{m}|{bg}"] = arr

        stats = group_stats_and_checks(factor_groups)
        if len(stats) < 2:
            return None

        # Test homogeneity of variances for between-subject factor groups at each model level.
        homogeneity_violated = False
        for m in models:
            model_data = [df_clean[(df_clean["Model"] == m) & (df_clean[between_factor] == bg)]["Score"].dropna().values for bg in between_groups]
            if len(model_data) > 1 and all(len(g) > 1 for g in model_data):
                stat_fligner, p_fligner = fligner(*model_data)
                if p_fligner < 0.05:
                    homogeneity_violated = True
                    break

        # Run mixed_anova.
        res = pg.mixed_anova(dv="Score", within="Model", correction='none',
                             between=between_factor, subject="SubjectID", data=df_clean)
        if res.empty:
            return None

        # Extract main effects and interaction.
        model_row = res[res["Source"] == "Model"]
        between_row = res[res["Source"] == between_factor]
        interaction_name = f"Model*{between_factor}"
        interaction_row = res[res["Source"] == interaction_name]

        # Initialize dictionary.
        row = {
            "Factor": factor,
            "Within": "Model",
            "Between": between_factor,
            "Test": "Mixed_ANOVA",
            "Homogeneity_Violated": homogeneity_violated
        }

        def add_effect_info(effect_row, prefix):
            # Adds F, p-value, and effect size if available.
            if not effect_row.empty:
                row[f"F_{prefix}"] = effect_row["F"].values[0]
                row[f"p_{prefix}"] = effect_row["p-unc"].values[0]
                if "np2" in effect_row.columns:
                    row[f"partial_eta_squared_{prefix}"] = effect_row["np2"].values[0]
                else:
                    row[f"partial_eta_squared_{prefix}"] = np.nan

        # Add model effect.
        add_effect_info(model_row, "Model")

        # Add between-subject effect.
        add_effect_info(between_row, between_factor)

        # Add interaction effect.
        if not interaction_row.empty:
            row["Interaction"] = interaction_name
            add_effect_info(interaction_row, "Interaction")

        # Add group-level information.
        idx = 1
        for k, st in stats.items():
            row[f"GroupName_{idx}"] = k
            row[f"N_{idx}"] = st["N"]
            row[f"Mean_{idx}"] = st["Mean"]
            row[f"Std_{idx}"] = st["Std"]
            row[f"Shapiro_p_{idx}"] = st["Shapiro_p"]
            row[f"Normal_{idx}"] = st["Normal"]
            idx += 1

        pd.DataFrame([row]).to_csv(filename, index=False)
        return True

    # 1. Run repeated-measures ANOVA by model.
    overall_path = os.path.join(anova_dir, "overall.csv")
    df_long = df_metric[["SubjectID", "Model", "Score"]].dropna()
    overall_res = perform_rm_anova(df_long, factor="Model")
    if overall_res is not None:
        pd.DataFrame([overall_res]).to_csv(overall_path, index=False)

    # 2. Run Welch's one-way ANOVA for each model by race and age.
    for model_val in df_metric["Model"].unique():
        model_subset = df_metric[df_metric["Model"] == model_val]
        model_anova_dir = os.path.join(anova_dir, model_val)
        os.makedirs(model_anova_dir, exist_ok=True)

        race_path = os.path.join(model_anova_dir, "race.csv")
        race_groups = {}
        for r_val in model_subset["Race"].unique():
            race_groups[r_val] = model_subset[model_subset["Race"] == r_val]["Score"].values
        race_res = perform_anova(race_groups, factor="Race")
        if race_res is not None:
            pd.DataFrame([race_res]).to_csv(race_path, index=False)

        age_path = os.path.join(model_anova_dir, "age.csv")
        age_groups = {}
        for a_val in model_subset["Age"].unique():
            age_groups[a_val] = model_subset[model_subset["Age"] == a_val]["Score"].values
        age_res = perform_anova(age_groups, factor="Age")
        if age_res is not None:
            pd.DataFrame([age_res]).to_csv(age_path, index=False)

    # 3. Run mixed-design ANOVA.
    race_mixed_path = os.path.join(anova_dir, "race.csv")
    perform_mixed_anova(df_metric, factor="Model", between_factor="Race", filename=race_mixed_path)

    age_mixed_path = os.path.join(anova_dir, "age.csv")
    perform_mixed_anova(df_metric, factor="Model", between_factor="Age", filename=age_mixed_path)

def create_disparity_heuristics(base_dir="results/analyze_colorization"):
    # Define patterns for race and age ANOVA results.
    race_pattern = os.path.join(base_dir, "*", "ANOVA_tests", "*", "race.csv")
    age_pattern = os.path.join(base_dir, "*", "ANOVA_tests", "*", "age.csv")
    
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
    os.makedirs(base_dir, exist_ok=True)
    
    # Save the results.
    race_heuristic_path = os.path.join(base_dir, "race_disparity_heuristic.csv")
    age_heuristic_path = os.path.join(base_dir, "age_disparity_heuristic.csv")
    race_df.to_csv(race_heuristic_path, index=False)
    age_df.to_csv(age_heuristic_path, index=False)

def create_disparity_barplots(base_dir="results/analyze_colorization"):
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
        create_disparity_heuristics(analysis_dir)
        create_disparity_barplots(analysis_dir)
