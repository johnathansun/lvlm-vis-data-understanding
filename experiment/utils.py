import pandas as pd
import time
from tqdm import tqdm
from openai import OpenAI
import anthropic
import os 
import random 
from concurrent.futures import ThreadPoolExecutor, as_completed
import re 
from ast import literal_eval
import numpy as np 

# os.environ["OPENAI_API_KEY"] = 
# os.environ["ANTHROPIC_API_KEY"] = 

MAX_RETRIES = 4
PARTIAL_SAVE_EVERY = 50              # save after every 5 tasks
BLANK_64 = "iVBORw0KGgoAAAANSUhEUgAAApQAAAHzCAYAAACe1o1DAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvc2/+5QAAAAlwSFlzAAAPYQAAD2EBqD+naQAACSRJREFUeJzt1sENwCAQwLDS/Xc+diAPhGRPkGfWzMwHAACH/tsBAAC8zVACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkGyJLgfivwmfiAAAAABJRU5ErkJggg=="
# ------------------------------------------------------------------------------

def experiment_to_prompt(x_vals, y_vals, output_format="dataframe"):
    """
    Converts the experiment details to a formatted prompt string.

    Parameters:
        x_vals: List of x values
        y_vals: List of y values
        output_format (str): One of "dataframe", "markdown", or "csv". Defaults to "dataframe".

    Returns:
        str: A string containing the experiment details formatted as specified.
    """
    # Zip the data into (x, y) pairs and shuffle the list to randomize the order.
    data_tuples = list(zip(x_vals, y_vals))
    random.seed(98)
    random.shuffle(data_tuples)

    # Unzip the shuffled tuples back into separate lists.
    shuffled_x, shuffled_y = zip(*data_tuples) if data_tuples else ([], [])

    if output_format == "dataframe":
        # Create a pandas DataFrame with shuffled data using 'x' and 'y' as columns.
        df = pd.DataFrame({
            "x": shuffled_x,
            "y": shuffled_y
        })
        return df.to_string(index=False)

    elif output_format == "markdown":
        # Format x and y values rounded to 2 decimal places.
        formatted_x = [f"{x:.2f}" for x in shuffled_x]
        formatted_y = [f"{y:.2f}" for y in shuffled_y]

        # Determine the maximum width required for each column.
        width_x = max(len("x"), max(len(x_str) for x_str in formatted_x))
        width_y = max(len("y"), max(len(y_str) for y_str in formatted_y))

        # Build header and divider rows with right alignment.
        header = f"| {'x'.rjust(width_x)} | {'y'.rjust(width_y)} |\n"
        divider = f"| {'-' * width_x} | {'-' * width_y} |\n"

        # Build the data rows, aligning each entry.
        rows = ""
        for x_str, y_str in zip(formatted_x, formatted_y):
            rows += f"| {x_str.rjust(width_x)} | {y_str.rjust(width_y)} |\n"
        return header + divider + rows

    elif output_format == "csv":
        # Build a CSV-like string with 'x' and 'y' as headers.
        header = "x,y\n"
        rows = ""
        for x, y in zip(shuffled_x, shuffled_y):
            # Format numbers rounded to 2 decimal places and replace internal commas.
            x_str = f"{x:.2f}" if isinstance(x, (int, float)) else str(x).replace(",", ";")
            y_str = f"{y:.2f}" if isinstance(y, (int, float)) else str(y).replace(",", ";")
            rows += f"{x_str},{y_str}\n"
        return header + rows
    else:
        raise ValueError("Unsupported output_format. Choose one of: 'dataframe', 'markdown', 'csv'.")

def round_to_significant_figures(x, sig=4):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.map(lambda v: round(v, sig - int(np.floor(np.log10(abs(v)))) - 1) if v != 0 else 0)
    elif isinstance(x, np.ndarray):
        return np.vectorize(lambda v: round(v, sig - int(np.floor(np.log10(abs(v)))) - 1) if v != 0 else 0)(x)
    else:
        return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0


def get_exp_response_from_b64(image_b64: str | None,
                              model: str, 
                              prompt_text: str) -> tuple[str, int, int]: # type: ignore
    """
    Send prompt (with optional image) to the specified model, retrying up to MAX_RETRIES times on failure.
    Returns (response_content, token_count).
    Raises the last exception if all retries fail.
    """
    # from img_utils import decode_base64_and_plot
    # decode_base64_and_plot(image_b64, False)
    # print(image_b64)

    OPENAI_MODELS = {"gpt-4.1-2025-04-14", "gpt-4o-2024-08-06"} # {"gpt-4o-2024-08-06"} 
    ANTHOPIC_MODELS = {"claude-3-5-sonnet-20241022"}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if model in OPENAI_MODELS and image_b64:
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    },
                                },
                                { "type": "text", "text": prompt_text },
                            ],
                        }
                    ]
                ) 
                return resp.choices[0].message.content, resp.usage.completion_tokens, resp.usage.prompt_tokens # type: ignore

            elif model in ANTHOPIC_MODELS and image_b64:
                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_b64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt_text
                                }
                            ],
                        },
                    ]
                )
                return resp.content[0].text, resp.usage.output_tokens, resp.usage.input_tokens # type: ignore
            
            elif model in OPENAI_MODELS: # no image 
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages = [
                        {"role": "user", "content": prompt_text}
                    ]
                )
                return resp.choices[0].message.content, resp.usage.completion_tokens, resp.usage.prompt_tokens # type: ignore
            
            elif model in ANTHOPIC_MODELS:
                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt_text}
                    ]
                )
                return resp.content[0].text, resp.usage.output_tokens, resp.usage.input_tokens # type: ignore
            
            else:
                raise ValueError(f"Model {model} not supported.")
        
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            else:
                # all retries exhausted
                return "!!!!ERROR!!!!", 0, 0

def get_llm_prompts(x_vals:list, y_vals:list,
                    img_b64: str,
                    blank_b64: str,
                    deceptive_b64: str|None,
                    data: bool = True, data_scatter: bool = True,
                    data_blank: bool = True, data_deceptive: bool=True, scatter: bool = True,
                    num_trials: int = 1, #! under new conditions, one trial per randomized
                    prompt_base: str = "What are patterns, trends, and features of interest in the following data?"
                ):
    """
    Goes from x_vals, y_vals, bank_b64, img_b64 to a list of tasks.
    """
    assert len(x_vals) == len(y_vals), "x_vals and y_vals must be the same length"
    assert num_trials > 0, "num_trials must be greater than 0"
    assert len(x_vals) > 0, "x_vals and y_vals must be non-empty"
    assert not (data_blank and blank_b64 is None), "blank_b64 must be provided if data_blank is True"
    assert not (data_scatter and img_b64 is None), "img_b64 must be provided if data_scatter is True"
    assert not (data_deceptive and deceptive_b64 is None), "deceptive_b64 must be provided if deceptive is True"

    prompt_data = experiment_to_prompt(x_vals, y_vals, output_format="dataframe")
    
    conditions = []
    if data:
        conditions.append(("data", f"{prompt_base}\n{prompt_data}", None))
    if data_scatter:
        conditions.append(("data-scatter", f"{prompt_base}\n{prompt_data}", img_b64))
    if data_blank:
        conditions.append(("data-blank", f"{prompt_base}\n{prompt_data}", blank_b64))
    if data_deceptive:
        conditions.append(("data-deceptive", f"{prompt_base}\n{prompt_data}", deceptive_b64))
    if scatter:
        conditions.append(("scatter", prompt_base, img_b64))
    
    all_tasks = []
    # Add tasks for this row
    for condition, prompt_text, img_b64 in conditions:
        for t in range(num_trials):
            all_tasks.append({
                "condition": condition, 
                "prompt_text": prompt_text, 
                "x_vals": x_vals,
                "y_vals": y_vals,
                "img_b64": img_b64,
                "deceptive_b64": deceptive_b64,
                "trial_idx": t,
            })
    
    return all_tasks


def process_row(row_idx, row, model_name, data: bool = True, data_scatter: bool = True, 
                data_blank: bool = True, data_deceptive: bool = True, scatter: bool = True):
    """Process a single DataFrame row: generate prompts and collect LLM responses."""
    results = []
    tasks = get_llm_prompts(
        x_vals=row["x"],
        y_vals=row["y"],
        img_b64=row.get("img_b64", None),
        blank_b64=BLANK_64,
        deceptive_b64=row.get("img_deceptive", None),
        prompt_base="What are patterns, trends, and features of interest in the following data?",
        data=data,
        data_scatter=data_scatter,
        data_blank=data_blank,
        data_deceptive=data_deceptive & (row.get("img_deceptive", None) is not None),
        scatter=scatter,
    )
    for task in tasks:
        # resp_text, tokens_used = "hi", 0 #!!! 
        resp_text, tokens_used, tokens_input = get_exp_response_from_b64(
            task["img_b64"],
            model_name,
            task["prompt_text"]
        )
        results.append({
            "row_idx": row_idx,
            "condition": task["condition"],
            "trial_idx": task["trial_idx"],
            "prompt": task["prompt_text"],
            "x_vals": task["x_vals"],
            "y_vals": task["y_vals"],
            "response": resp_text,
            "tokens_used": tokens_used, 
            "tokens_input": tokens_input,
            "model": model_name,
        })
    return results

def run_trial_threaded(df_filepath, model_name, max_workers=4, 
                       head:None|int=None, random:bool|None=None, 
                       data: bool = True, data_scatter: bool = True,
                       data_blank: bool = True, data_deceptive: bool = True, scatter: bool = True):
    # Setup trial folder
    parent_folder = os.path.dirname(df_filepath)
    existing = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    numbers = [int(re.fullmatch(r"trial_(\d+)", d).group(1)) 
               for d in existing if re.fullmatch(r"trial_(\d+)", d)]
    next_trial = max(numbers) + 1 if numbers else 0
    trial_folder = os.path.join(parent_folder, f"trial_{next_trial}")
    os.makedirs(trial_folder, exist_ok=False)
    
    # Load and rehydrate data
    df_input = pd.read_csv(df_filepath) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if random and head: 
        df_input = df_input.sample(frac=head/len(df_input)).reset_index(drop=True)
    if head and not random: 
        df_input = df_input.head(head) if head else df_input

    df_input["x"] = df_input["x"].apply(literal_eval)
    df_input["y"] = df_input["y"].apply(literal_eval)

    all_results = []
    futures = []

    # Thread over rows
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in df_input.iterrows():
            futures.append(executor.submit(process_row, idx, row, model_name, data, data_scatter, data_blank, data_deceptive, scatter))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Rows processed"):
            row_results = future.result()
            # print(row_results)
            all_results.extend(row_results)
            # Partial save
            if len(all_results) % PARTIAL_SAVE_EVERY == 0:
                pd.DataFrame(all_results).to_csv(
                    os.path.join(trial_folder, "results_partial.csv"),
                    index=False
                )

    # Final save
    df_input.to_csv(os.path.join(trial_folder, "input_dataframe.csv"), index=False)
    pd.DataFrame(all_results).to_csv(
        os.path.join(trial_folder, "results.csv"),
        index=False
    )

    if len(df_filepath) > PARTIAL_SAVE_EVERY and len(df_input) > PARTIAL_SAVE_EVERY:
        os.remove(os.path.join(trial_folder, "results_partial.csv"))

    return all_results
