import io
import sys 
sys.path.append("/Users/victoriali/Documents/GitHub/llms-vis/")
sys.path.append("/Users/johnathansun/Documents/llms-vis/")
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from public.experiment.utils import run_trial_threaded, round_to_significant_figures
from public.experiment.img_utils import hash_image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM experiments on datasets.")

    #! REPLACE THIS LINE
    filepath = "/your/file/path/input_dataframe.csv"

    if filepath is None: 
        parser.add_argument(
            "filepath", 
            type=str, 
            required=True, 
            help="List of file paths to process"
        )
        filepaths = parser.parse_args().filepaths
    else: 
        print("Going based on provided filepaths")

    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="Name of the model to use (e.g., gpt-4.1-2025-04-14, claude-3-5-sonnet-20241022)."
    )

    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=8, 
        help="Number of samples to process."
    )

    run_trial_threaded(filepath, model_name=parser.parse_args().model_name, 
                       max_workers=parser.parse_args().max_workers) # , data_blank=False, data=False 

#     example usage 
#     python generate_responses.py --model_name claude-3-5-sonnet-20241022 --max_workers 8
#     python generate_responses.py --model_name gpt-4.1-2025-04-14 --max_workers 12
