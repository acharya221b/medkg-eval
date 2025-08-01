# main.py
import os
import pandas as pd
import argparse
import logging
import ast
import json
import glob
from tqdm import tqdm

# --- CRITICAL: Import all necessary components from the correct files ---
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

from RAG_pipeline.generator import RAGGenerator
from RAG_pipeline.utils import load_faiss_index, connect_nebula, MODEL_NAME
from evaluation.evaluator import FullDataEval
from evaluation.utils import clean_output # Assuming this is in the root evaluation folder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TASK_TO_PROMPT_MAP = {'reasoning_fct': 'reasoning_fct', 'reasoning_fake': 'reasoning_fake', 'reasoning_nota': 'reasoning_nota'}

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

def run_prediction_for_model(args, model_name, generator):
    for task_name in args.tasks:
        output_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{model_name}.csv"
        output_path = os.path.join(args.predictions_dir, output_filename)
        if os.path.exists(output_path) and not args.force_rerun:
            logging.info(f"Skipping task '{task_name}' for model '{model_name}', file exists.")
            continue
        input_csv = os.path.join(args.data_dir, f"{task_name}.csv")
        df = pd.read_csv(input_csv)
        prompt_assets = load_prompt_assets(TASK_TO_PROMPT_MAP[task_name], args.prompt_id, args.max_shots)
        predictions = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Predicting for {task_name} with {model_name}"):
            try:
                options = ast.literal_eval(row['options'])
                output = generator.predict(row['question'], options, prompt_assets, task_name, args.no_rag)
                predictions.append({'id': row['id'], 'output': str(output)})
            except Exception as e:
                logging.error(f"Error on id {row['id']}: {e}")
                predictions.append({'id': row['id'], 'output': "{}"})
        pd.DataFrame(predictions).to_csv(output_path, index=False, header=True)
        logging.info(f"Predictions saved to {output_path}")

def run_json_conversion_stage(args):
    for model_name in args.models:
        for task_name in args.tasks:
            pred_filename = f"{task_name}_predictions_prompt_{args.prompt_id}_model_{model_name}.csv"
            pred_path = os.path.join(args.predictions_dir, pred_filename)
            if not os.path.exists(pred_path): continue
            dataset_path = os.path.join(args.data_dir, f"{task_name}.csv")
            merge_df = pd.merge(pd.read_csv(dataset_path), pd.read_csv(pred_path), on='id')
            merge_df["gpt_output"] = merge_df.apply(lambda row: clean_output(row['id'], row['output']), axis=1)
            if 'correct_index' in merge_df.columns:
                merge_df['testbed_data'] = merge_df.apply(lambda r: {'correct_index': r['correct_index']}, axis=1)
            else:
                merge_df['testbed_data'] = [{} for _ in range(len(merge_df))]
            json_filename = f"{task_name}_prompt_{args.prompt_id}_model_{model_name}.json"
            json_path = os.path.join(args.results_dir, json_filename)
            merge_df[['id', 'testbed_data', 'gpt_output']].to_json(json_path, orient='records', indent=4)
            logging.info(f"Converted predictions to JSON: {json_path}")

def run_evaluation_stage(args):
    logging.info(f"--- Starting Final Consolidated Evaluation ---")
    json_pattern = f"*_prompt_{args.prompt_id}_model_*.json"
    eval_simple = FullDataEval(args.results_dir, file_pattern=json_pattern, correct_score=1, incorrect_score=0).run_all_evaluations()
    eval_penalty = FullDataEval(args.results_dir, file_pattern=json_pattern, correct_score=1, incorrect_score=-0.25).run_all_evaluations()
    if eval_simple.empty:
        logging.error("Evaluation produced no results.")
        return
    eval_simple.rename(columns={'score': 'simple_score'}, inplace=True)
    eval_penalty.rename(columns={'score': 'penalty_score'}, inplace=True)
    final_df = pd.merge(eval_simple, eval_penalty[['task_name', 'model_name', 'penalty_score']], on=['task_name', 'model_name'], how='left')
    final_df = final_df[final_df['task_name'].isin(args.tasks)]
    final_df.sort_values(by=['task_name', 'model_name'], inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    report_path = os.path.join(args.results_dir, f"final_report_prompt_{args.prompt_id}_tasks_{'_'.join(args.tasks)}.csv")
    final_df.to_csv(report_path, index=False)
    print("\n--- Final Evaluation Report ---")
    print(final_df.to_string())

def main(args):
    """The main execution function, now aware of the --no-rag flag."""
    st_model, faiss_index, faiss_texts, nebula_client, nebula_pool = None, None, None, None, None
    try:
        # --- CONDITIONAL LOADING OF HEAVY RESOURCES ---
        if not args.no_rag:
            logging.info("--- RAG mode enabled. Loading all shared resources... ---")
            st_model = SentenceTransformer(MODEL_NAME)
            faiss_index, faiss_texts = load_faiss_index()
            nebula_client, nebula_pool = connect_nebula()
            if st_model is None or faiss_index is None or nebula_client is None:
                raise RuntimeError("One or more critical RAG resources failed to load.")
        else:
            logging.info("--- No-RAG mode enabled. Skipping resource loading. ---")

        if not args.skip_predictions:
            for model in args.models:
                try:
                    # Initialize the generator, passing RAG components (or None if in no-RAG mode)
                    rag_generator = RAGGenerator(model, st_model, faiss_index, faiss_texts, nebula_client)
                    run_prediction_for_model(args, model, rag_generator)
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
    parser.add_argument("--models", nargs='+', required=True, help="List of model names.")
    parser.add_argument("--tasks", nargs='+', required=True, choices=TASK_TO_PROMPT_MAP.keys(), help="List of tasks.")
    parser.add_argument("--prompt_id", type=str, default="v0", help="Prompt ID.")
    parser.add_argument("--max_shots", type=int, default=3, help="Max few-shot examples.")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--predictions_dir", type=str, default="predictions", help="Predictions directory.")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory.")
    parser.add_argument("--skip_predictions", action="store_true", help="Skip prediction generation.")
    parser.add_argument("--no-rag", action="store_true", help="Skip the entire RAG pipeline and query the LLM directly.")
    parser.add_argument("--force_rerun", action="store_true", help="Force regeneration of predictions.")
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