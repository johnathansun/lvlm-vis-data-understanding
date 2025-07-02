# %%
import time 
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from jinja2 import Environment
from typing import Type, Any, Dict, Tuple
from pydantic import BaseModel
from openai import OpenAI
from google import genai
import time

PARTIAL_SAVE_EVERY = 50
MAX_RETRIES = 4

gemini_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) 

def gemini_judge(
    prompt: str,
    output_format: Type[BaseModel],
    model: str = "gemini-2.5-flash"
) -> Tuple[Any, Any]:
    """
    Calls the LLM with retries and parses the response.
    """ # "gemini-2.0-flash", 
    assert model in {"gemini-2.5-pro", "gemini-2.5-flash"}, "must use supported model"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = gemini_client.models.generate_content(
                model=model,
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": output_format,
                    },
            )
            return resp, resp.parsed
        except Exception as e:
            print(f"[gemini_judge] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)  # backoff before retry
            else:
                print("[gemini_judge] All retries failed. Returning (None, None).")
                return None, None
    return None, None

def process_judge_row(
    row_idx: int,
    row: pd.Series,
    template,
    output_format: Type[BaseModel],
) -> Dict[str, Any]:
    context = row.to_dict()
    prompt  = template.render(**context)
    response, event = gemini_judge(prompt, output_format=output_format)
    
    result: Dict[str, Any] = {"row_idx": row_idx, **context}
    if event is not None:
        result.update(event.dict())
    result["raw_llm_text"] = getattr(response, "output", None)
    return result

def run_llm_judge(
    results_csv_path:      str,
    default_output_format: Type[BaseModel],
    default_template_path: str,
    special_output_format: Type[BaseModel],
    special_template_path: str,
    special_condition:     str    = "data-deceptive",
    max_workers:           int    = 4,
    partial_save_every:    int    = 50
) -> pd.DataFrame:
    """
    Judge each row in results_csv_path, using one template/format for
    'normal' rows and a second template/format for rows where
    df['condition']==special_condition.
    """
    # prep output folder
    results_folder = os.path.dirname(results_csv_path)
    trial_name     = os.path.basename(results_folder)
    parent_folder  = os.path.dirname(results_folder)

    judge_root = os.path.join(parent_folder, "judge")
    os.makedirs(judge_root, exist_ok=True)
    out_folder = os.path.join(judge_root, trial_name)
    # try:
    os.makedirs(out_folder, exist_ok=True)
    # except FileExistsError:
    #     raise RuntimeError(f"Won't overwrite existing judge folder: {out_folder!r}")

    # load cs
    df = pd.read_csv(results_csv_path)

    # read both templates from disk
    with open(default_template_path, 'r') as f:
        default_src = f.read()
    with open(special_template_path, 'r') as f:
        special_src = f.read()

    env = Environment()
    default_tpl = env.from_string(default_src)
    special_tpl = env.from_string(special_src)

    # spawn threads, picking template+format per row
    all_judged = []
    futures     = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for idx, row in df.iterrows():
            if row.get("condition") == special_condition:
                tpl   = special_tpl
                ofmt  = special_output_format
            else:
                tpl   = default_tpl
                ofmt  = default_output_format

            futures.append(
                exe.submit(
                    process_judge_row,
                    idx, row, tpl, ofmt
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            judged = fut.result()
            all_judged.append(judged)

            if len(all_judged) % partial_save_every == 0:
                pd.DataFrame(all_judged).to_csv(
                    os.path.join(out_folder, "gemini_judge_results_partial.csv"),
                    index=False
                )

    # final save & cleanup
    final_df = pd.DataFrame(all_judged)
    final_df.to_csv(os.path.join(out_folder, "gemini_judge_results.csv"), index=False)
    partial = os.path.join(out_folder, "gemini_judge_results_partial.csv")
    if os.path.exists(partial):
        os.remove(partial)

    return final_df
