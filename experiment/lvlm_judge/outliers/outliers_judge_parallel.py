from public.experiment.judge import run_llm_judge
from pydantic import BaseModel
import argparse 
import pandas as pd

WORKING_PATH = "your_working_path"

class OutlierResponse(BaseModel):
    identifies_any_outlier: bool
    identifies_correct_outlier: bool
    identifies_other_outliers: bool
    reasoning: str

class DeceptiveResponse(BaseModel):
    identifies_discrepancy: bool
    identifies_any_outlier: bool
    identifies_correct_outlier: bool
    identifies_other_outliers: bool
    reasoning: str

def modify_for_prompt(trial_num):
    assert trial_num in [4,5], "the ones that were run"
    import pandas as pd 
    from ast import literal_eval

    def fn_to_apply(x,y):
        # Create a pandas DataFrame with shuffled data using 'x' and 'y' as columns.
        df = pd.DataFrame({
            "x": x,
            "y": y
        })
        return df.to_string(index=False)
    # trial_num = 5
    filepath = WORKING_PATH + f"{trial_num}/results.csv"
    info_path = WORKING_PATH + f"{trial_num}/input_dataframe.csv"
    df = pd.read_csv(filepath)
    print(df.columns)
    cols = df.columns.tolist()
    info = pd.read_csv(info_path)
    df = pd.merge(df, info, left_on=["x_vals", "y_vals"], right_on=["x","y"], how="outer")
    df["x_vals"] = df["x_vals"].apply(literal_eval)
    df["y_vals"] = df["y_vals"].apply(literal_eval)
    df["dataframe"] = df.apply(lambda row: fn_to_apply(row["x_vals"], row["y_vals"]), axis=1)
    # in outlier change [ to ( and ] to )
    df["outlier"] = df["outlier"].apply(lambda x: x.replace("[", "(").replace("]", ")"))
    cols.extend(["outlier", "dataframe"])
    df = df[cols]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM experiments on datasets.")

    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=10, 
        help="Number of samples to process."
    )
    results_path = f"{WORKING_PATH}/results.csv"

    print(f"ANALYZING: {results_path}")
    run_llm_judge(
        results_csv_path=results_path,
        default_output_format=OutlierResponse,  # Replace with actual model
        default_template_path=f"{WORKING_PATH}/outliers_judge_prompt.j2",
        special_output_format=DeceptiveResponse,  # Replace with actual model
        special_template_path=f"{WORKING_PATH}/outliers_wrong_judge_prompt.j2",
        special_condition="data-deceptive",
        max_workers=16,
        partial_save_every=50
    )
