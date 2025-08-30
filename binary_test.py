import logging
import json
import os
import ast
import sys
import pandas as pd
from dotenv import load_dotenv
from json_repair import repair_json
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from RAG_pipeline.utils import (
    load_faiss_index, connect_nebula, retrieve_semantic_nodes,
    get_definitions_from_graph, rerank_definitions, generate_llm_response
)

# --- 0. SCRIPT SETUP ---

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Load environment variables from .env file
env_path = './.env'
if not os.path.exists(env_path):
    logging.error(f".env file not found at {env_path}. Please create it with your API keys and configuration.")
    sys.exit(1)
load_dotenv(dotenv_path=env_path)

logging.info("Initial setup and environment variables loaded.")

# --- 1. CORE FUNCTIONS & CONFIGURATION ---

def load_prompt_assets(task_name, prompt_id, max_shots, library_dir="prompt_library"):
    """Loads prompt templates, output formats, and few-shot examples from a library."""
    assets = {"prompt": "", "output_format": "", "shots": []}
    task_dir = os.path.join(library_dir, task_name)

    if not os.path.isdir(library_dir) or not os.path.isdir(task_dir):
        logging.warning(f"Prompt directory not found at {task_dir}. Proceeding without few-shot examples.")
        return assets

    prompts_path = os.path.join(task_dir, "prompts.json")
    if os.path.exists(prompts_path):
        with open(prompts_path, 'r') as f:
            try:
                prompts = json.load(f).get("prompts", [])
                selected = next((p for p in prompts if p.get("id") == prompt_id), None)
                if selected:
                    assets.update(selected)
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {prompts_path}")

    shots_path = os.path.join(task_dir, "shots.json")
    if os.path.exists(shots_path):
        with open(shots_path, 'r') as f:
            try:
                shots_list = json.load(f).get("shots", [])
                loaded_shots = shots_list[0] if shots_list and isinstance(shots_list[0], list) else shots_list
                assets["shots"] = loaded_shots[:max_shots]
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {shots_path}")

    logging.info(f"Loaded {len(assets['shots'])} shots for task '{task_name}' using prompt '{prompt_id}'.")
    return assets

# --- Model & File Configuration ---
_CURR_DIR = os.getcwd()
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
INDEX_FILE = os.path.join(_CURR_DIR, "../graph_rag/faiss_index.bin")
TEXTS_FILE = os.path.join(_CURR_DIR, "../graph_rag/semantic_nodes.json")

# --- Debugging & RAG Parameters ---
TASK_NAME = 'reasoning_fct'
PROMPT_ID = 'v3'
MAX_SHOTS = 3
MODEL_NAME_TO_DEBUG = 'deepseek-r1:14b'
QUESTION_ID_TO_DEBUG = "0ac6c5c7-9826-441a-81d5-68478e6299bb" #"839de867-3100-4283-a219-ec349eee415f" #"140d832a-b8ae-4791-aada-6fd62f313adb" 
RETRIEVAL_TOP_K = 30000
RERANK_TOP_K = 15
no_rag_flag = True
# --- 2. LOAD ASSETS & DATA ---

try:
    if not no_rag_flag:
        st_model = SentenceTransformer(MODEL_NAME)
        faiss_index, faiss_texts = load_faiss_index()
        nebula_client, nebula_pool = connect_nebula()
    llm_client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("API_KEY"))
    logging.info("Successfully loaded models, FAISS index, and connected to NebulaGraph.")
except Exception as e:
    logging.error(f"Failed to initialize models or clients: {e}")
    sys.exit(1)

data_file = f"data/{TASK_NAME}.csv"
if not os.path.exists(data_file):
    logging.error(f"Data file not found at {data_file}. Please ensure the path is correct.")
    sys.exit(1)

df = pd.read_csv(data_file)
question_rows = df[df['id'] == QUESTION_ID_TO_DEBUG]

if question_rows.empty:
    logging.error(f"Question ID '{QUESTION_ID_TO_DEBUG}' not found in {data_file}.")
    sys.exit(1)

question_row = question_rows.iloc[0]

try:
    question = question_row['question']
    options = ast.literal_eval(question_row['options'])
except (KeyError, SyntaxError) as e:
    logging.error(f"Failed to parse question data for ID '{QUESTION_ID_TO_DEBUG}': {e}")
    sys.exit(1)

print(f"\n--- DEBUGGING ID: {QUESTION_ID_TO_DEBUG} ---")
print(f"Question: {question}")
print(f"Options: {options}")
print("--------------------------------------------------\n")

# --- 3. EXECUTE THE RAG PIPELINE (ONCE) ---

final_context = []

if not no_rag_flag:
    logging.info(f"--- STAGE 1: Semantic Retrieval (Top {RETRIEVAL_TOP_K}) ---")
    query = f"{question} {' '.join(options.values())}"
    suis, top_semantic_texts = retrieve_semantic_nodes(query, st_model, faiss_index, faiss_texts, top_k=RETRIEVAL_TOP_K, top_m=30)
    logging.info(f"Retrieved {len(top_semantic_texts)} semantic context documents.")
    print("--------------------------------------------------\n")

    logging.info("--- STAGE 2: Knowledge Graph Traversal ---")
    graph_definitions = get_definitions_from_graph(nebula_client, suis)
    logging.info(f"Retrieved {len(graph_definitions)} definitions from the graph.")
    print("--------------------------------------------------\n")

    logging.info(f"--- STAGE 3: Re-ranking (Top {RERANK_TOP_K}) ---")
    final_definitions = rerank_definitions(question, graph_definitions, top_k=RERANK_TOP_K)
    logging.info(f"Re-ranked to the top {len(final_definitions)} most relevant context documents.")
    final_context = list(set(top_semantic_texts + final_definitions))
    logging.info(f"Combined semantic and graph contexts into {len(final_context)} unique documents.")
    
    print("--------------------------------------------------\n")

# --- 4. LLM GENERATION AND PROBABILITY CALCULATION ---
logging.info("--- STAGE 4: LLM Generation (Looping for Probability) ---")
prompt_assets = load_prompt_assets(TASK_NAME, PROMPT_ID, MAX_SHOTS, library_dir="prompt_library")

# Initialize counters for the experiment
yes_count = 0
no_count = 0
invalid_runs_count = 0
total_runs = 50

for i in range(total_runs):
    logging.info(f"--- Running iteration {i + 1}/{total_runs} ---")
    
    llm_output = generate_llm_response(
        llm_client,
        MODEL_NAME_TO_DEBUG,
        question,
        options,
        final_context,
        prompt_assets,
        no_rag=no_rag_flag
    )

    if llm_output and 'is_answer_correct' in llm_output:
        # Get the value, convert to string, lowercase and strip whitespace for robust comparison
        is_correct_val = str(llm_output.get('is_answer_correct')).lower().strip()
        #retrieved_answer = str(llm_output.get('answer')).lower().strip()
        #logging.info(f"Iteration {i + 1}: Answer was {retrieved_answer}")
        if is_correct_val == 'yes':
            yes_count += 1
            logging.info(f"Iteration {i + 1}: Response was 'yes'")
        elif is_correct_val == 'no':
            no_count += 1
            logging.info(f"Iteration {i + 1}: Response was 'no'")
        else:
            invalid_runs_count += 1
            logging.warning(f"Iteration {i + 1}: Invalid value for 'is_answer_correct': {llm_output.get('is_answer_correct')}")
    else:
        invalid_runs_count += 1
        logging.warning(f"Iteration {i + 1}: LLM response missing 'is_answer_correct' key or failed to generate.")

# --- 5. FINAL ANALYSIS AND SUMMARY ---
print("\n\n--- EXPERIMENT SUMMARY ---")

# Calculate probabilities
if total_runs > 0:
    probability_yes = (yes_count / total_runs) * 100
    probability_no = (no_count / total_runs) * 100
else:
    probability_yes = 0
    probability_no = 0

print(f"Question ID:               {QUESTION_ID_TO_DEBUG}")
print(f"Total Runs:                 {total_runs}")
print(f"Count of 'Yes' responses:   {yes_count}")
print(f"Count of 'No' responses:    {no_count}")
print(f"Invalid/Failed Runs:        {invalid_runs_count}")
print("---------------------------------")
print(f"Probability of 'Yes':       {probability_yes:.2f}%")
print(f"Probability of 'No':        {probability_no:.2f}%")
print("---------------------------------")

# Cleanly close the connection pool
if nebula_pool:
    nebula_pool.close()
    logging.info("NebulaGraph connection pool closed.")