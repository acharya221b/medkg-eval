import logging
import json
import os
import ast
import sys
import pandas as pd
from dotenv import load_dotenv
import asyncio

# --- Centralized Imports from your project ---
from RAG_pipeline.utils import (
    load_faiss_index, 
    connect_nebula, 
    retrieve_semantic_nodes,
    get_definitions_from_graph, 
    rerank_definitions,
    generate_llm_response, # <-- Import this
    format_shots,         # <-- Import this
    check_premise_consistency,
    MODEL_NAME
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import AsyncOpenAI

# --- 0. SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
load_dotenv(dotenv_path='./.env')
logging.info("Initial setup and environment variables loaded.")

# --- 1. CONFIGURATION ---
TASK_NAME = 'reasoning_fct'
PROMPT_ID = 'v3'
MAX_SHOTS = 3
MODEL_NAME_TO_DEBUG = 'deepseek-r1:14b'
QUESTION_ID_TO_DEBUG = "0ac6c5c7-9826-441a-81d5-68478e6299bb"
RETRIEVAL_TOP_K = 30000
RERANK_TOP_K = 15

def load_prompt_assets(task_name, prompt_id, max_shots, library_dir="prompt_library"):
    # This function is fine to keep here as it's specific to loading prompts for tests.
    assets = {"prompt": "", "output_format": "", "shots": []}
    task_dir = os.path.join(library_dir, task_name)
    if not os.path.isdir(library_dir) or not os.path.isdir(task_dir): return assets
    prompts_path = os.path.join(task_dir, "prompts.json")
    if os.path.exists(prompts_path):
        with open(prompts_path, 'r') as f:
            try:
                prompts = json.load(f).get("prompts", [])
                selected = next((p for p in prompts if p.get("id") == prompt_id), None)
                if selected: assets.update(selected)
            except json.JSONDecodeError: logging.error(f"Error decoding JSON from {prompts_path}")
    shots_path = os.path.join(task_dir, "shots.json")
    if os.path.exists(shots_path):
        with open(shots_path, 'r') as f:
            try:
                shots_list = json.load(f).get("shots", [])
                loaded_shots = shots_list[0] if shots_list and isinstance(shots_list[0], list) else shots_list
                assets["shots"] = loaded_shots[:max_shots]
            except json.JSONDecodeError: logging.error(f"Error decoding JSON from {shots_path}")
    logging.info(f"Loaded {len(assets['shots'])} shots for task '{task_name}' using prompt '{prompt_id}'.")
    return assets


async def main():
    """Main asynchronous function to run the entire test."""
    # --- 2. LOAD ASSETS & DATA ---
    st_model = SentenceTransformer(MODEL_NAME)
    cross_encoder = CrossEncoder('pritamdeka/S-PubMedBert-MS-MARCO')
    faiss_index, faiss_texts = load_faiss_index()
    nebula_client, nebula_pool = connect_nebula() # Assumes this returns only the pool
    llm_client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("API_KEY"))
    
    if any(v is None for v in [st_model, cross_encoder, faiss_index, faiss_texts, nebula_pool, llm_client]):
        logging.error("One or more critical resources failed to load. Exiting.")
        sys.exit(1)
    
    logging.info("Successfully loaded all models and clients.")

    df = pd.read_csv(f"data/{TASK_NAME}.csv")
    question_row = df[df['id'] == QUESTION_ID_TO_DEBUG].iloc[0]
    question = question_row['question']
    options = ast.literal_eval(question_row['options'])

    print(f"\n--- DEBUGGING ID: {QUESTION_ID_TO_DEBUG} ---")

    # --- 3. EXECUTE THE RAG PIPELINE (ONCE) ---
    no_rag_flag = False
    final_context = []

    if not no_rag_flag:
        logging.info("--- STAGE 1: Semantic Retrieval ---")
        query = f"{question} {' '.join(options.values())}"
        suis, top_semantic_texts = await retrieve_semantic_nodes(query, st_model, faiss_index, faiss_texts, top_k=RETRIEVAL_TOP_K, top_m=30)
        
        logging.info("--- STAGE 2: Knowledge Graph Traversal ---")
        # --- FIXED: Pass the nebula_pool, not nebula_client ---
        graph_definitions = await get_definitions_from_graph(nebula_pool, suis)
        
        logging.info("--- STAGE 3: Re-ranking and Combination ---")
        
        combined_context = await rerank_definitions(cross_encoder, question, graph_definitions, top_k=RERANK_TOP_K)
        final_context = list(set(top_semantic_texts + combined_context))
        logging.info(f"RAG pipeline complete. Final context size: {len(final_context)} documents.")
        #context_str_for_check = " ".join(final_context)
        #consistency_result = await check_premise_consistency(llm_client, MODEL_NAME_TO_DEBUG, question, context_str_for_check)

    # --- 4. LLM GENERATION LOOP ---
    logging.info("--- STAGE 4: Calculating Probability via Forced-Choice Loop ---")
    prompt_assets = load_prompt_assets(TASK_NAME, PROMPT_ID, MAX_SHOTS)

    # --- Use asyncio.gather for a massive speedup on the loop ---
    async def get_forced_choice():
        # IMPORTANT: We use the IMPORTED async version of generate_llm_response
        return await generate_llm_response(
            llm_client=llm_client, model_name=MODEL_NAME_TO_DEBUG,
            question=question, options=options, definitions=final_context,
            prompt_assets=prompt_assets, no_rag=no_rag_flag,
            mode='forced_choice'
        )

    total_runs = 20
    tasks = [get_forced_choice() for _ in range(total_runs)]
    results = await asyncio.gather(*tasks)

    yes_count = results.count('yes')
    no_count = results.count('no')
    error_count = total_runs - (yes_count + no_count)

    # --- 5. FINAL ANALYSIS AND SUMMARY ---
    print("\n\n--- EXPERIMENT SUMMARY ---")
    valid_runs = yes_count + no_count
    probability_yes = (yes_count / valid_runs) * 100 if valid_runs > 0 else 0
    probability_no = (no_count / valid_runs) * 100 if valid_runs > 0 else 0

    print(f"Total Runs:                 {total_runs}")
    print(f"Count of 'Yes' responses:   {yes_count}")
    print(f"Count of 'No' responses:    {no_count}")
    print(f"Invalid/Error Runs:         {error_count}")
    print("---------------------------------")
    print(f"Probability of 'Yes':       {probability_yes:.2f}%")
    print(f"Probability of 'No':        {probability_no:.2f}%")
    print("---------------------------------")

    if nebula_pool:
        nebula_pool.close()
        logging.info("NebulaGraph connection closed.")

if __name__ == "__main__":
    asyncio.run(main())