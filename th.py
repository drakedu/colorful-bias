import os
import shutil

# Base directory, assuming you're running this script from the project base directory.
base_dir = os.getcwd()
analysis_dir = os.path.join(base_dir, "results", "analyze_colorization")

# Iterate over each metric directory inside results/analyze_colorization
if os.path.isdir(analysis_dir):
    for metric in os.listdir(analysis_dir):
        metric_dir = os.path.join(analysis_dir, metric)
        
        if os.path.isdir(metric_dir):
            # Paths to old directories
            anova_dir = os.path.join(metric_dir, "ANOVA")
            kruskal_dir = os.path.join(metric_dir, "kruskalwallis_test")

            # Remove ANOVA directory if it exists
            if os.path.isdir(anova_dir):
                print(f"Removing directory: {anova_dir}")
                shutil.rmtree(anova_dir)

            # Remove kruskalwallis_test directory if it exists
            if os.path.isdir(kruskal_dir):
                print(f"Removing directory: {kruskal_dir}")
                shutil.rmtree(kruskal_dir)
else:
    print(f"Directory does not exist: {analysis_dir}")
