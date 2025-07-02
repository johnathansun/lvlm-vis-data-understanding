import sys 
sys.path.append("/Users/victoriali/Documents/GitHub/llms-vis/")
sys.path.append("/Users/johnathansun/Documents/llms-vis/")
from public.experiment.judge import run_llm_judge
from pydantic import BaseModel
import argparse 
import pandas as pd

WORKING_DIR = "YOUR-WORKING-DIR"

class ClusterResponse(BaseModel):
    identifies_any_cluster: bool
    num_clusters: int
    reasoning: str

class DeceptiveClusterResponse(BaseModel):
    identifies_any_cluster: bool
    identifies_discrepancy: bool
    num_clusters_data : int
    num_clusters_scatter : int
    reasoning: str

def modify_for_prompt(trial_num):
    import pandas as pd 
    from ast import literal_eval

    def fn_to_apply(x,y):
        df = pd.DataFrame({
            "x": x,
            "y": y
        })
        return df.to_string(index=False)
    # trial_num = 5
    filepath = f"{WORKING_DIR}/{trial_num}/results.csv"
    info_path = f"{WORKING_DIR}/{trial_num}/input_dataframe.csv"
    df = pd.read_csv(filepath)
    print(df.columns)
    cols = df.columns.tolist()
    info = pd.read_csv(info_path)
    df = pd.merge(df, info, left_on=["x_vals", "y_vals"], right_on=["x","y"], how="outer")
    df["x_vals"] = df["x_vals"].apply(literal_eval)
    df["y_vals"] = df["y_vals"].apply(literal_eval)
    df["dataframe"] = df.apply(lambda row: fn_to_apply(row["x_vals"], row["y_vals"]), axis=1)
    # in outlier change [ to ( and ] to )
    # df["num_clusters"] = df["outlier"].apply(lambda x: x.replace("[", "(").replace("]", ")"))
    cols.extend(["num_clusters", "dataframe"])
    df = df[cols]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM experiments on datasets.")

    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=10, 
        help="Number of samples to process."
    )
    results_path = f"{WORKING_DIR}/results.csv" # some trial number

    print(f"ANALYZING: {results_path}")
    run_llm_judge(
        results_csv_path=results_path,
        default_output_format=ClusterResponse,
        default_template_path=f"{WORKING_DIR}/clusters_judge_prompt.j2",
        special_output_format=DeceptiveClusterResponse,
        special_template_path=f"{WORKING_DIR}/clusters_wrong_judge_prompt.j2",
        special_condition="data-deceptive",
        max_workers=16,
        partial_save_every=50
    )
