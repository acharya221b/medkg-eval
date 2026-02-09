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
import time 
# --- CRITICAL: Import all necessary components from the correct files ---
import faiss
import numpy as np
from sentence_transformers import models, SentenceTransformer, CrossEncoder
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

from RAG_pipeline.generator import RAGGenerator
from RAG_pipeline.utils import load_faiss_index, connect_nebula, MODEL_NAME
from evaluation.evaluator import FullDataEval
from evaluation.utils import clean_output # Assuming this is in the root evaluation folder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ALL_TASKS = ['reasoning_fct', 'reasoning_fake', 'reasoning_nota', 
             'IR_pmid2title', 'IR_title2pubmedlink', 'IR_abstract2pubmedlink', 'IR_pubmedlink2title']

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

        df = pd.read_csv(os.path.join(args.data_dir, f"{task_name}.csv"))
        library_dir = "prompt_library/IR_RAG" if task_name.startswith('IR_') and not args.no_rag else "prompt_library"
        prompt_assets = load_prompt_assets(TASK_TO_PROMPT_MAP[task_name], args.prompt_id, args.max_shots, args.no_rag, library_dir=library_dir)
        semaphore = asyncio.Semaphore(10)
        predictions = []

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


async def run_single_task_prediction_async(args, model_name, task_name, generator):
    """This is a simplified version of your original function that runs one task."""
    sanitized_model_name = model_name.replace('/', '_')
    output_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{sanitized_model_name}.csv"
    output_path = os.path.join(args.predictions_dir, output_filename)
    
    df = pd.read_csv(os.path.join(args.data_dir, f"{task_name}.csv"))
    library_dir = "prompt_library" # Always default for no-rag
    prompt_assets = load_prompt_assets(TASK_TO_PROMPT_MAP[task_name], args.prompt_id, args.max_shots, args.no_rag, library_dir=library_dir)
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # We pass None for context because this is the no-rag path
    tasks = [process_row(row, generator, prompt_assets, task_name, args.no_rag, semaphore) 
             for _, row in df.iterrows()]
    
    predictions_outputs = await tqdm_asyncio.gather(*tasks, desc=f"Predicting for {task_name} with {model_name} (No RAG)")
    
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


def load_prompt_assets(task_name, prompt_id, max_shots, no_rag, library_dir="prompt_library"):
    assets = {"prompt": "", "output_format": "", "shots": []}
    task_dir = os.path.join(library_dir, task_name)
    prompts_path = os.path.join(task_dir, "prompts.json")
    if os.path.exists(prompts_path):
        with open(prompts_path, 'r') as f:
            prompts = json.load(f).get("prompts", [])
            selected = next((p for p in prompts if p.get("id") == prompt_id), None)
            if selected: assets.update(selected)
    shots_file = "no_rag_shots.json" if no_rag and task_name=="reasoning_fct" else "shots.json"
    shots_path = os.path.join(task_dir, shots_file)
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
    """
    Runs the final evaluation stage.
    MODIFIED: Now dynamically searches for JSON files only for the tasks
    specified in the --tasks argument, across all models and prompt IDs.
    """
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


async def main_async(args):
    """The main asynchronous execution function, with the corrected logical fork."""
    st_model, cross_encoder, faiss_index, faiss_texts, nebula_pool = None, None, None, None, None
    try:

        if args.no_rag:
            # --- PATH 1: NO-RAG WORKFLOW ---
            logging.info("--- Starting pipeline in NO-RAG mode ---")
            for model_name in args.models:
                # --- NEW: CHECK FOR EXISTING PREDICTIONS ---
                sanitized_model_name = model_name.replace('/', '_')
                # This loop is now inside the model loop
                for task_name in args.tasks:
                    output_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{sanitized_model_name}.csv"
                    output_path = os.path.join(args.predictions_dir, output_filename)
                    
                    if os.path.exists(output_path) and not args.force_rerun:
                        logging.info(f"Predictions for '{model_name}' on task '{task_name}' already exist. Skipping.")
                        continue # Skip to the next task
                
                    try:
                        # We only need one generator per model
                        rag_generator = RAGGenerator(model_name, st_model, cross_encoder, faiss_index, faiss_texts, nebula_pool)
                        # We now call a modified function that processes one task at a time
                        await run_single_task_prediction_async(args, model_name, task_name, rag_generator)
                        #await run_prediction_for_model_async(args, model_name, rag_generator)
                    except Exception as e:
                        logging.critical(f"FATAL: Generator for model {model_name} on task {task_name} failed. Error: {e}")

        else:

            # --- PATH 2: RAG WORKFLOW (Efficient "Retrieve Once") ---
            logging.info("--- Starting pipeline in RAG mode ---")
            nebula_pool = connect_nebula()
            if nebula_pool is None:
                raise RuntimeError("Nebula Pool is required for all RAG tasks but failed to load.")
            
            for task_name in args.tasks:
                logging.info(f"--- Processing Task: {task_name} for all models ---")

                models_to_run = []
                if not args.force_rerun:
                    for model_name in args.models:
                        sanitized_model_name = model_name.replace('/', '_')
                        output_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{sanitized_model_name}.csv"
                        output_path = os.path.join(args.predictions_dir, output_filename)
                        if not os.path.exists(output_path):
                            models_to_run.append(model_name)
                        else:
                            logging.info(f"Predictions for '{model_name}' on task '{task_name}' already exist. Skipping.")
                else:
                    models_to_run = args.models # If force_rerun, run all models.
                
                # 2. If there are no new models to run for this task, skip the ENTIRE task.
                if not models_to_run:
                    logging.info(f"All model predictions for task '{task_name}' already exist. Skipping entire task.")
                    continue

                # --- STAGE 1: RETRIEVAL (with Caching) ---
                full_df = pd.read_csv(os.path.join(args.data_dir, f"{task_name}.csv"))                
                cache_dir = "retrieval_cache"
                os.makedirs(cache_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d")
                cache_file = os.path.join(cache_dir, f"{task_name}_context_cache_{timestamp}.json")
                #cache_file = os.path.join(cache_dir, "reasoning_nota_context_cache_20251016.json")
                retrieved_contexts = []
                if os.path.exists(cache_file) and not args.get_context:
                    logging.info(f"Found existing context cache. Loading from: {cache_file}")
                    with open(cache_file, 'r') as f:
                        retrieved_contexts = json.load(f)
                else:    
                    logging.info(f"--- Starting Retrieval Stage for {len(full_df)} records ---")
                    needs_semantic_search = any(not task.startswith('IR_') for task in args.tasks)
                    if needs_semantic_search:
                        logging.info("--- Reasoning task detected. Loading semantic search components... ---")
                        st_model = SentenceTransformer(MODEL_NAME)
                        cross_encoder = CrossEncoder('ncbi/MedCPT-Cross-Encoder')
                        faiss_index, faiss_texts = load_faiss_index()
                        if st_model is None or faiss_index is None:
                            raise RuntimeError("Semantic search components failed to load.")
                    else:
                        logging.info("--- Only IR tasks detected. Skipping semantic search components. ---")
                    retrieval_generator = RAGGenerator(args.models[0], st_model, cross_encoder, faiss_index, faiss_texts, nebula_pool)

                    retrieval_semaphore = asyncio.Semaphore(10)
                    retrieval_tasks = []
                    for index, row in full_df.iterrows():
                        async def retrieve_with_semaphore(row_data):
                            async with retrieval_semaphore:
                                try:
                                    input_key = None
                                    output_key = None
                                    if task_name.startswith('IR_'):
                                        input_key = TASK_TO_INPUT_KEY_MAP.get(task_name, '')
                                        output_key = TASK_TO_OUTPUT_KEY_MAP.get(task_name, '')
                                        question_text = str(row_data.get(input_key, ''))
                                        input_key = input_key.lower()
                                        output_key = output_key.lower()
                                        options_dict = {}
                                    else:
                                        question_text = str(row_data.get('question', ''))
                                        options_dict = ast.literal_eval(row_data.get('options', '{}'))
                                    
                                    return await retrieval_generator.retrieve_context_only(question_text, options_dict, task_name, input_key, output_key)
                                except Exception as e:
                                        logging.error(f"Failed to parse 'options' for row index {index} in task '{task_name}'. Error: {e}. Skipping retrieval.")
                                        return []
                        
                        # 3. Append the helper coroutine to the tasks list.
                        retrieval_tasks.append(retrieve_with_semaphore(row))

                    retrieved_contexts = await tqdm_asyncio.gather(*retrieval_tasks, desc=f"Retrieving context for {task_name}")
                    logging.info(f"Saving retrieved context to cache: {cache_file}")
                    with open(cache_file, 'w') as f:
                        json.dump(retrieved_contexts, f)
                    
                # --- STAGE 2: GENERATION (Run for each model using the cached context) ---
                for model_name in models_to_run:
                    logging.info(f"--- Generating predictions for model: {model_name} on task: {task_name} ---")

                    #if retrieved_contexts is None or len(retrieved_contexts) != len(full_df):
                    if retrieved_contexts is None:
                        generator = RAGGenerator(model_name, st_model, cross_encoder, faiss_index, faiss_texts, nebula_pool)
                    else:
                        generator = RAGGenerator(model_name, None, None, None, None, nebula_pool)
                    sanitized_model_name = model_name.replace('/', '_')
                    output_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{sanitized_model_name}.csv"
                    output_path = os.path.join(args.predictions_dir, output_filename)


                    library_dir = "prompt_library/IR_RAG" if task_name.startswith('IR_') else "prompt_library"
                    prompt_assets = load_prompt_assets(TASK_TO_PROMPT_MAP[task_name], args.prompt_id, args.max_shots, args.no_rag, library_dir=library_dir)
                    semaphore = asyncio.Semaphore(args.concurrency)
                    
                    prediction_tasks = []
                    for i, (_, row) in enumerate(full_df.iterrows()):
                        # Define a small async helper to manage the semaphore for prediction
                        async def predict_with_semaphore(row_data, context):
                            async with semaphore:
                                row_id = row_data.get('id', 'UNKNOWN_ID')
                                try:
                                    input_key = None
                                    output_key = None
                                    if task_name.startswith('IR_'):
                                        input_key = TASK_TO_INPUT_KEY_MAP[task_name]
                                        output_key = TASK_TO_OUTPUT_KEY_MAP[task_name]
                                        question_text = str(row_data.get(input_key, ""))
                                        input_key = input_key.lower()
                                        output_key = output_key.lower()
                                        options_dict = {}
                                    else:
                                        question_text = str(row_data.get('question', ""))
                                        options_string = row_data.get('options', '{}')
                                        if pd.isna(options_string): # Handle potential NaN values
                                            options_string = '{}'
                                        options_dict = ast.literal_eval(options_string)

                                    
                                    output = await generator.predict(
                                        question=question_text, options=options_dict, prompt_assets=prompt_assets,
                                        task_name=task_name, no_rag=args.no_rag, context=context,
                                        input_key=input_key,
                                        output_key=output_key
                                    )
                                    return (row_id, output)
                                except Exception as e:
                                    logging.error(f"Failed to parse 'options' for row {row_id} during generation. Error: {e}. Skipping prediction.")
                                    return (row_id, None)
                        
                        prediction_tasks.append(predict_with_semaphore(row, retrieved_contexts[i]))
                    
                    predictions_outputs = await tqdm_asyncio.gather(*prediction_tasks, desc=f"Predicting for {task_name} with {model_name}")
                    
                    predictions = [{'id': row_id, 'output': json.dumps(output) if output else "{}"} for row_id, output in predictions_outputs]
                    pd.DataFrame(predictions).to_csv(output_path, index=False, header=True)
                    logging.info(f"Predictions for {model_name} saved to {output_path}")

        # --- COMMON FINAL STAGES ---
        # These functions run at the end, regardless of the mode.
        run_json_conversion_stage(args)
        run_evaluation_stage(args)

    finally:
        if nebula_pool:
            nebula_pool.close()
        logging.info("NebulaGraph connection closed.")

def main(args):
    """The main entry point, now calling the async main function."""
    if args.eval_only:
        logging.info("--- Starting pipeline in EVALUATION-ONLY mode ---")
        logging.info("Skipping all retrieval and generation stages.")
        
        # Directly call the final two stages.
        run_json_conversion_stage(args)
        run_evaluation_stage(args)
        
        logging.info("--- Evaluation-only run finished ---")
        return
    asyncio.run(main_async(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical KG Evaluation Pipeline.")
    parser.add_argument("--models", nargs='+', help="List of model names.")
    parser.add_argument("--tasks", nargs='+', required=True, choices=TASK_TO_PROMPT_MAP.keys(), help="List of tasks.")
    parser.add_argument("--prompt_id", type=str, default="v0", help="Prompt ID.")
    parser.add_argument("--max_shots", type=int, default=3, help="Max few-shot examples.")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--predictions_dir", type=str, default="predictions", help="Predictions directory.")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory.")
    parser.add_argument("--no-rag", action="store_true", help="Skip the entire RAG pipeline and query the LLM directly.")
    parser.add_argument("--force_rerun", action="store_true", help="Force regeneration of predictions.")
    parser.add_argument("--get_context", action="store_true", help="Force regeneration of retrieval context.")
    parser.add_argument("--eval_only", action="store_true", help="Skip all retrieval and generation, run only the final evaluation stages.")
    parser.add_argument("--subset_size", type=int, default=100, help="Number of records per evaluation subset.")
    # parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    # parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate.")
    # parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling.")
    parser.add_argument("--concurrency",type=int,default=10, help="concurrency limit set by asyncio.Semaphore")
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