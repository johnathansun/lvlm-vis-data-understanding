import pandas as pd 
from ast import literal_eval
import re 
import numpy as np 

# BASE = "/Users/vrli/Desktop/"
BASE = "/Users/johnathansun/Documents/public/"

########################################################
## Clusters 
########################################################

def get_clusters_df(success_analysis:bool=True):
    cluster_params_df = pd.read_csv(f'{BASE}data/clusters/input_dataframe.csv')
    gpt_cluster_df = pd.read_csv(f'{BASE}data/clusters/gpt_gemini_judge_results.csv')
    claude_cluster_df = pd.read_csv(f'{BASE}data/clusters/claude_gemini_judge_results.csv')

    gpt_cluster_df = gpt_cluster_df.rename(columns={"num_clusters": "num_clusters_judge"})
    claude_cluster_df = claude_cluster_df.rename(columns={"num_clusters": "num_clusters_judge"})

    claude_cluster_df = claude_cluster_df.merge(cluster_params_df, left_on=["x_vals", "y_vals"], right_on=["x", "y"], how="outer")
    gpt_cluster_df = gpt_cluster_df.merge(cluster_params_df, left_on=["x_vals", "y_vals"], right_on=["x", "y"], how="outer")

    cluster_df = pd.concat([claude_cluster_df, gpt_cluster_df], axis=0).reset_index(drop=True)

    cluster_df['x_vals'] = cluster_df['x_vals'].apply(literal_eval)
    cluster_df['y_vals'] = cluster_df['y_vals'].apply(literal_eval)

    # different prompt for data-deceptive
    for i in range(len(cluster_df)):
        row = cluster_df.iloc[i]
        if row['condition'] == 'data-deceptive':
            cluster_proposal = None
            cluster_proposal = row['num_clusters_data']
            cluster_df.at[i, 'num_clusters_judge'] = cluster_proposal

    cluster_df["num_clusters_judge"] = cluster_df["num_clusters_judge"].apply(lambda x: 0 if x == -1 else x).astype(int)
    cluster_df["num_clusters_judge"] = cluster_df["num_clusters_judge"].astype(int)

    if success_analysis: 
        cluster_df["success"] = cluster_df.apply(lambda x: x['num_clusters_judge'] == x['num_clusters'], axis=1)
        cluster_df = cluster_df.groupby(['num_clusters', 'condition', 'model']).agg({
            'success': 'sum'
        }).reset_index()
    else:
        # summary analysis & supplementary materials
        cluster_df.rename(columns={"num_clusters": "Subtlety -- True Number of Clusters", "num_clusters_judge": "Model Response Classification"}, inplace=True)
        cluster_df["setting"] = "cluster"
    return cluster_df

########################################################
## Nonlinearities 
########################################################

def get_labels(df: pd.DataFrame):
    specific = ["parabola", "parabolic", "quadratic",
                "concave", "opens down"]

    broad = ["nonlinear", "non-linear",
            r"not\s+\w+\s+linear", r"isn't\s+\w+\s+linear",
            "isn't linear", "not linear",
            "curved", "polynomial"]

    def detect_patterns(text, patterns):
        if pd.isna(text):
            return 0
        standardized_text = re.sub(r"[^\w\s]", "", text).lower()
        return int(any(re.search(pattern, standardized_text) for pattern in patterns))

    df["specific_detected"] = df["response"].apply(lambda x: detect_patterns(x, specific))
    df["broad_detected"] = df["response"].apply(lambda x: detect_patterns(x, broad))
    
    return df

def get_nonlinearity_df(success_analysis:bool=True): 
    nonlinearity_params_df = pd.read_csv(f'{BASE}data/nonlinearity/input_dataframe.csv')
    gpt_nonlinearity_df = pd.read_csv(f'{BASE}data/nonlinearity/gpt_results.csv')
    claude_nonlinearity_df = pd.read_csv(f'{BASE}data/nonlinearity/claude_results.csv')
        
    claude_nonlinearity_df = claude_nonlinearity_df.merge(nonlinearity_params_df, left_on=["x_vals", "y_vals"], right_on=["x", "y"], how="outer")
    gpt_nonlinearity_df = gpt_nonlinearity_df.merge(nonlinearity_params_df, left_on=["x_vals", "y_vals"], right_on=["x", "y"], how="outer")

    nonlinearity_df = pd.concat([gpt_nonlinearity_df, claude_nonlinearity_df], axis=0)
    nonlinearity_df = get_labels(nonlinearity_df)

    if success_analysis:
        nonlinearity_df = nonlinearity_df.groupby(["model", "condition", "extra_points"]).agg({
            "specific_detected": "sum",
            "broad_detected": "sum"
            }).reset_index()
    else: 
        # summary analysis & supplementary materials
        nonlinearity_df['label'] = np.where((nonlinearity_df['specific_detected'] == 1) & (nonlinearity_df['broad_detected'] == 1), "Both", 
                               np.where((nonlinearity_df['specific_detected'] == 1) & (nonlinearity_df['broad_detected'] == 0), "Specific Only", 
                               np.where((nonlinearity_df['specific_detected'] == 0) & (nonlinearity_df['broad_detected'] == 1), "Broad Only", "Neither"))) 
        nonlinearity_df.rename(columns={"extra_points": "Subtlety -- Number of Points Beyond Vertex", "label": "Model Response Classification"}, inplace=True)
        nonlinearity_df["setting"] = "nonlinearity"
    return nonlinearity_df 

########################################################
## Outliers 
########################################################

def get_outlier_df(success_analysis:bool=True): 
    gpt_outlier_df = pd.read_csv(f"{BASE}data/outlier/gpt_gemini_judge_results.csv") # z scores < 2
    claude_outlier_df = pd.read_csv(f"{BASE}data/outlier/claude_gemini_judge_results.csv") # z scores >= 2
    info_df = pd.read_csv(f"{BASE}data/outlier/outlier_lines_400.csv")
    gpt_outlier_df = pd.merge(gpt_outlier_df, info_df, left_on=["x_vals", "y_vals"], right_on=["x", "y"], how="outer")
    claude_outlier_df = pd.merge(claude_outlier_df, info_df, left_on=["x_vals", "y_vals"], right_on=["x", "y"], how="outer")
    
    outlier_df = pd.concat([gpt_outlier_df, claude_outlier_df], axis=0, ignore_index=True) 
    outlier_df["z_score"] = outlier_df["type"].replace("<=1", "≤1")

    if success_analysis:
        outlier_df["success"] = outlier_df["identifies_any_outlier"] & outlier_df["identifies_correct_outlier"]
        outlier_df = outlier_df.groupby(['z_score', 'condition', 'model']).agg({
            'success': 'sum'
        }).reset_index()
    else: 
        # summary analysis & supplementary materials
        outlier_df["label"] = np.where((outlier_df["identifies_correct_outlier"] == 1) & (outlier_df["identifies_other_outliers"] == 0), "Correct\nOutlier\nOnly", 
                                     np.where(outlier_df["identifies_other_outliers"] == 1, "Other/\nMultiple\nOutliers", "No\nOutlier"))
        
        outlier_df.rename(columns={"z_score": "Subtlety -- Vertical Z-Score", "label": "Model Response Classification"}, inplace=True)
        outlier_df["setting"] = "outlier"
    
    return outlier_df
