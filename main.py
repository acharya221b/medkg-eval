# main.py
import os
import sys
import pandas as pd
import argparse
import logging
import ast
import json
import glob
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# --- CRITICAL: Import all necessary components from the correct files ---
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

from RAG_pipeline.generator import RAGGenerator
from RAG_pipeline.utils import load_faiss_index, connect_nebula, MODEL_NAME
from evaluation.evaluator import FullDataEval
from evaluation.utils import clean_output # Assuming this is in the root evaluation folder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ALL_TASKS = ['reasoning_fct', 'reasoning_fake', 'reasoning_nota', 
             'IR_pmid2title', 'IR_title2url', 'IR_abstract2pubmedlink', 'IR_pubmedlink2title']

TASK_TO_PROMPT_MAP = {
    'reasoning_fct': 'reasoning_fct', 
    'reasoning_fake': 'reasoning_fake', 
    'reasoning_nota': 'reasoning_nota',
    # --- ADD: Map IR tasks to their prompt library folders ---
    'IR_pmid2title': 'IR_pmid2title',
    'IR_title2pubmedlink': 'IR_title2pubmedlink',
    'IR_abstract2pubmedlink': 'IR_abstract2pubmedlink',
    'IR_pubmedlink2title': 'IR_pubmedlink2title'
}

TASK_TO_INPUT_KEY_MAP = {
    'IR_pmid2title': 'PMID',
    'IR_title2pubmedlink': 'Title',
    'IR_abstract2pubmedlink': 'Abstract',
    'IR_pubmedlink2title': 'url'
}

TASK_TO_OUTPUT_KEY_MAP = {
    "IR_pmid2title": "Title",
    "IR_pubmedlink2title": "Title",
    "IR_title2pubmedlink": "url",
    "IR_abstract2pubmedlink": "url"
}


def process_row_sync_for_debug(row, generator, prompt_assets, task_name, no_rag):
    # This function waits for the semaphore before running the prediction
    row_id = row.get('id', 'UNKNOWN_ID')
    try:
        if task_name.startswith('IR_'):
            input_key = TASK_TO_INPUT_KEY_MAP[task_name]
            output_key = TASK_TO_OUTPUT_KEY_MAP[task_name]
            question_text = str(row.get(input_key, ""))

            output_dict = generator.predict(
                question=question_text,
                prompt_assets=prompt_assets,
                task_name=task_name,
                no_rag=no_rag,
                input_key=input_key.lower(),
                output_key=output_key.lower()
            )
        else:
            question_text = str(row.get('question', ""))
            options_dict = ast.literal_eval(row.get('options', '{}'))
            output_dict = generator.predict(
                question=question_text,
                options=options_dict,
                prompt_assets=prompt_assets,
                task_name=task_name,
                no_rag=no_rag,
            )
            
        return (row_id, output_dict)
    except Exception as e:
        logging.error(f"Error on id {row_id}: {e}", exc_info=True)
        return (row_id, None)
        
def run_prediction_for_model_sync_for_debug(args, model_name, generator):
    for task_name in args.tasks:
        # ... (file path and loading logic is the same)
        output_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{model_name}.csv"
        output_path = os.path.join(args.predictions_dir, output_filename)
        #...
        df = pd.read_csv(os.path.join(args.data_dir, f"{task_name}.csv"))
        library_dir = "prompt_library/IR_RAG" if task_name.startswith('IR_') and not args.no_rag else "prompt_library"
        prompt_assets = load_prompt_assets(TASK_TO_PROMPT_MAP[task_name], args.prompt_id, args.max_shots, library_dir=library_dir)

        logging.info(f"--- RUNNING IN SYNC DEBUG MODE ---")
        predictions = []
        # Use a standard tqdm progress bar for easy debugging
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Debugging {task_name}"):
            # CALL THE SYNCHRONOUS WRAPPER
            row_id, output = process_row_sync_for_debug(row, generator, prompt_assets, task_name, args.no_rag)
            predictions.append({'id': row_id, 'output': json.dumps(output) if output else "{}"})

        # Save results at the end
        pd.DataFrame(predictions).to_csv(output_path, index=False, header=True)
        logging.info(f"Predictions saved to {output_path}")

async def process_row(row, generator, prompt_assets, task_name, no_rag, semaphore):
    # This function waits for the semaphore before running the prediction
    async with semaphore:
        try:
            row_id = row.get('id', 'UNKNOWN_ID')
            if task_name.startswith('IR_'):
                input_key = TASK_TO_INPUT_KEY_MAP[task_name]
                output_key = TASK_TO_OUTPUT_KEY_MAP[task_name]
                question_text = str(row.get(input_key, ""))

                output_dict = await generator.predict(
                    question=question_text,
                    prompt_assets=prompt_assets,
                    task_name=task_name,
                    no_rag=no_rag,
                    input_key=input_key.lower(),
                    output_key=output_key.lower()
                )
            else:
                question_text = str(row.get('question', ""))
                options_dict = ast.literal_eval(row.get('options', '{}'))
                output_dict = await generator.predict(
                    question=question_text,
                    options=options_dict,
                    prompt_assets=prompt_assets,
                    task_name=task_name,
                    no_rag=no_rag
                )
                
            return (row_id, output_dict)
        except Exception as e:
            logging.error(f"Error on id {row_id}: {e}", exc_info=True)
            return (row_id, None)
        
async def run_prediction_for_model_async(args, model_name, generator):
    for task_name in args.tasks:
        sanitized_model_name = model_name.replace('/', '_')
        output_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{sanitized_model_name}.csv"
        output_path = os.path.join(args.predictions_dir, output_filename)
        #...
        df = pd.read_csv(os.path.join(args.data_dir, f"{task_name}.csv"))
        library_dir = "prompt_library/IR_RAG" if task_name.startswith('IR_') and not args.no_rag else "prompt_library"
        prompt_assets = load_prompt_assets(TASK_TO_PROMPT_MAP[task_name], args.prompt_id, args.max_shots, library_dir=library_dir)
        semaphore = asyncio.Semaphore(10)
        predictions = []
        # Use a standard tqdm progress bar for easy debugging
        # for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Debugging {task_name}"):
        #     # CALL THE SYNCHRONOUS WRAPPER
        #     row_id, output = process_row(row, generator, prompt_assets, task_name, args.no_rag, semaphore)
        #     predictions.append({'id': row_id, 'output': json.dumps(output) if output else "{}"})
        # generation_params = {
        #     "temperature": args.temperature,
        #     "max_tokens": args.max_new_tokens,
        #     "top_p": args.top_p,
        # }
        tasks = [process_row(row, generator, prompt_assets, task_name, args.no_rag, semaphore) 
                 for _, row in df.iterrows()]
        logging.info(f"Executing {len(tasks)} predictions for task '{task_name}'...")
        predictions_outputs = await tqdm_asyncio.gather(*tasks, desc=f"Predicting for {task_name} with {model_name}")
        
        # Process the results
        predictions = []
        for i, output in predictions_outputs:
            #row_id = df.iloc[i]['id']
            if output is None:
                logging.error(f"Error on id {i}: Prediction returned None")
                predictions.append({'id': i, 'output': "{}"})
            else:
                # Ensure the output is a string representation of the dict/JSON
                predictions.append({'id': i, 'output': json.dumps(output)})
        
        pd.DataFrame(predictions).to_csv(output_path, index=False, header=True)
        logging.info(f"Predictions saved to {output_path}")


def load_prompt_assets(task_name, prompt_id, max_shots, library_dir="prompt_library"):
    assets = {"prompt": "", "output_format": "", "shots": []}
    task_dir = os.path.join(library_dir, task_name)
    prompts_path = os.path.join(task_dir, "prompts.json")
    if os.path.exists(prompts_path):
        with open(prompts_path, 'r') as f:
            prompts = json.load(f).get("prompts", [])
            selected = next((p for p in prompts if p.get("id") == prompt_id), None)
            if selected: assets.update(selected)
    shots_path = os.path.join(task_dir, "shots.json")
    if os.path.exists(shots_path):
        with open(shots_path, 'r') as f:
            shots_list = json.load(f).get("shots", [])
            loaded_shots = shots_list[0] if shots_list and isinstance(shots_list[0], list) else shots_list
            assets["shots"] = loaded_shots[:max_shots]
    logging.info(f"Loaded {len(assets['shots'])} shots for '{task_name}' (max_shots: {max_shots}).")
    return assets


def run_json_conversion_stage(args):
    """
    Converts CSV predictions to a JSON format, now with targeted ground truth
    for IR tasks and robust parsing.
    """
    for model_name in args.models:
        for task_name in args.tasks:
            sanitized_model_name = model_name.replace('/', '_')
            pred_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{sanitized_model_name}.csv"
            pred_path = os.path.join(args.predictions_dir, pred_filename)
            if not os.path.exists(pred_path):
                logging.warning(f"Prediction file not found, skipping: {pred_path}")
                continue
                
            dataset_path = os.path.join(args.data_dir, f"{task_name}.csv")
            if not os.path.exists(dataset_path):
                logging.warning(f"Dataset file not found, skipping: {dataset_path}")
                continue

            df_dataset = pd.read_csv(dataset_path)
            df_preds = pd.read_csv(pred_path)
            merge_df = pd.merge(df_dataset, df_preds, on='id')

            # --- This is the key fix for the empty gpt_output ---
            # `clean_output` (in utils.py) must now correctly parse the JSON string.
            #merge_df['output'] = merge_df['output'].astype(str).fillna('{}') # Ensure it's a string, handle NaNs
            merge_df["gpt_output"] = merge_df.apply(lambda row: clean_output(row['id'], row['output']), axis=1)

            # --- Logic to populate 'testbed_data' based on task type ---
            if task_name.startswith('IR_'):
                # This logic is now TARGETED.
                logging.debug(f"Handling IR task '{task_name}' for JSON conversion.")
                
                # 1. Get the name of the column that holds the correct answer.
                ground_truth_col = TASK_TO_OUTPUT_KEY_MAP.get(task_name)
                
                if not ground_truth_col or ground_truth_col not in merge_df.columns:
                    logging.error(f"Configuration error or missing column! Ground truth column '{ground_truth_col}' not found for task '{task_name}'.")
                    # Create an empty testbed_data if the column is missing
                    merge_df['testbed_data'] = [{}] * len(merge_df)
                else:
                    # 2. Create the testbed_data dictionary with a consistent key.
                    # The evaluator for IR tasks (evaluate_ir_task) looks for keys like 'Title' and 'url'.
                    # We will use the column name as the key.
                    merge_df['testbed_data'] = merge_df.apply(
                        lambda row: {ground_truth_col: row[ground_truth_col]}, axis=1
                    )
            
            else:
                # Your original, working logic for reasoning tasks remains untouched.
                logging.debug(f"Handling Reasoning task '{task_name}' for JSON conversion.")
                if 'correct_index' in merge_df.columns:
                    merge_df['testbed_data'] = merge_df.apply(lambda r: {'correct_index': r['correct_index']}, axis=1)
                else:
                    merge_df['testbed_data'] = [{} for _ in range(len(merge_df))]
            
            # Save to JSON
            json_filename = f"{task_name}_prompt_{args.prompt_id}_model_{sanitized_model_name}.json"
            json_path = os.path.join(args.results_dir, json_filename)
            merge_df[['id', 'testbed_data', 'gpt_output']].to_json(json_path, orient='records', indent=4)
            logging.info(f"Converted predictions to JSON: {json_path}")

def run_evaluation_stage(args):
    logging.info(f"--- Starting Final Consolidated Evaluation ---")
    json_pattern = f"*_prompt_{args.prompt_id}_model_*.json"
    evaluator = FullDataEval(
        args.results_dir, 
        file_pattern=json_pattern, 
        correct_score=1, 
        incorrect_score=-0.25
    )
    final_df = evaluator.run_all_evaluations()

    if final_df.empty:
        logging.error("Evaluation produced no results.")
        return
        
    # 2. Rename the 'score' column to be specific.
    #    The 'correct' column already serves the purpose of the old 'simple_score'.
    final_df.rename(columns={'score': 'penalty_score'}, inplace=True)
    
    # 3. Filter, sort, and clean up the DataFrame as before.
    final_df = final_df[final_df['task_name'].isin(args.tasks)]
    final_df.sort_values(by=['task_name', 'model_name'], inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    report_path = os.path.join(args.results_dir, f"final_report_prompt_{args.prompt_id}_tasks_{'_'.join(args.tasks)}.csv")
    final_df.to_csv(report_path, index=False)
    print("\n--- Final Evaluation Report ---")
    print(final_df.to_string())

def main(args):
    """The main execution function, now aware of the --no-rag flag."""
    st_model, cross_encoder, faiss_index, faiss_texts, nebula_pool = None, None, None, None, None
    try:
        # --- CONDITIONAL LOADING OF HEAVY RESOURCES ---
        # --- MODIFIED: SMART RESOURCE LOADING ---
        if args.no_rag:
            logging.info("--- No-RAG mode enabled. Skipping all resource loading. ---")
        else:
            # Step 1: Check if any Reasoning tasks are requested.
            # Reasoning tasks require the full semantic search stack.
            needs_semantic_search = any(not task.startswith('IR_') for task in args.tasks)

            # Step 2: All RAG tasks (both IR and Reasoning) require a Nebula connection.
            logging.info("--- RAG mode enabled. Connecting to NebulaGraph... ---")
            _, nebula_pool = connect_nebula()
            if nebula_pool is None:
                raise RuntimeError("Nebula Pool is required for all RAG tasks but failed to load.")
            
            # Step 3: Conditionally load the heavy components ONLY if needed.
            if needs_semantic_search:
                logging.info("--- Reasoning task detected. Loading semantic search components... ---")
                #st_model = SentenceTransformer(MODEL_NAME)
                #cross_encoder = CrossEncoder('pritamdeka/S-PubMedBert-MS-MARCO')
                st_model = SentenceTransformer(MODEL_NAME, device='cpu')
                cross_encoder = CrossEncoder('pritamdeka/S-PubMedBert-MS-MARCO', device='cpu')
                faiss_index, faiss_texts = load_faiss_index()
                if st_model is None or faiss_index is None:
                     raise RuntimeError("Semantic search components failed to load.")
            else:
                logging.info("--- Only IR tasks detected. Skipping loading of SentenceTransformer, CrossEncoder, and Faiss index. ---")

        if not args.skip_predictions:
            for model in args.models:
                try:
                    # Initialize the generator, passing RAG components (or None if in no-RAG mode)
                    rag_generator = RAGGenerator(model, st_model, cross_encoder, faiss_index, faiss_texts, nebula_pool)
                    #asyncio.run(run_prediction_for_model_async(args, model, rag_generator))
                    asyncio.run(run_prediction_for_model_async(args, model, rag_generator))
                except Exception as e:
                    logging.critical(f"FATAL: Generator for model {model} failed. Error: {e}")
                    continue
        
        run_json_conversion_stage(args)
        run_evaluation_stage(args)
    finally:
        if nebula_pool:
            nebula_pool.close()
            logging.info("NebulaGraph connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical KG Evaluation Pipeline.")
    parser.add_argument("--models", nargs='+', help="List of model names.")
    parser.add_argument("--tasks", nargs='+', required=True, choices=TASK_TO_PROMPT_MAP.keys(), help="List of tasks.")
    parser.add_argument("--prompt_id", type=str, default="v0", help="Prompt ID.")
    parser.add_argument("--max_shots", type=int, default=3, help="Max few-shot examples.")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--predictions_dir", type=str, default="predictions", help="Predictions directory.")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory.")
    parser.add_argument("--skip_predictions", action="store_true", help="Skip prediction generation.")
    parser.add_argument("--no-rag", action="store_true", help="Skip the entire RAG pipeline and query the LLM directly.")
    parser.add_argument("--force_rerun", action="store_true", help="Force regeneration of predictions.")
    # parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    # parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate.")
    # parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling.")
    # parser.add_argument("--batch_size",type=int,default=10, help="concurrency limit set by asyncio.Semaphore")
    args = parser.parse_args()
    if args.no_rag:
        args.predictions_dir = "predictions_no_rag"
        args.results_dir = "results_no_rag"
    else:
        # Keep default names if RAG is enabled
        args.predictions_dir = args.predictions_dir
        args.results_dir = args.results_dir
    os.makedirs(args.predictions_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    main(args)
    logging.info(f"--- Full Pipeline Finished ---")