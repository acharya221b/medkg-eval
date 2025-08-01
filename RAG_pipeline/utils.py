# RAG_pipeline/utils.py
import logging
import json
import re
import time
import os
from dotenv import load_dotenv
from json_repair import repair_json
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

# --- ROBUST FILE PATHS ---
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# --- THE DEFINITIVE FIX IS HERE ---
# This path correctly navigates up two levels from utils.py to find the graph_rag directory.
INDEX_FILE = os.path.join(_CURR_DIR, "../../graph_rag/faiss_index.bin")
TEXTS_FILE = os.path.join(_CURR_DIR, "../../graph_rag/semantic_nodes.json")

# --- ALL HELPER FUNCTIONS NOW LIVE IN THIS FILE ---

def connect_nebula():
    try:
        config = Config()
        config.max_connection_pool_size = 10
        connection_pool = ConnectionPool()
        connection_pool.init([("127.0.0.1", 9669)], config)
        client = connection_pool.get_session("root", "nebula")
        client.execute("USE petagraph;")
        logging.info("Successfully connected to NebulaGraph.")
        return client, connection_pool
    except Exception as e:
        logging.error(f"Failed to connect to NebulaGraph: {e}")
        return None, None

def load_faiss_index():
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(TEXTS_FILE, "r") as f: texts = json.load(f)
        logging.info(f"FAISS index loaded with {index.ntotal} vectors.")
        return index, texts
    except Exception as e:
        logging.error(f"Could not load FAISS index or texts file: {e}")
        return None, None

def retrieve_semantic_seeds(query, model, index, texts, top_k=30000):
    query_vec = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    return [texts[i]["sui"] for i in indices[0]]

def get_definitions_from_graph(client, suis):
    if not suis: return []
    try:
        suis_str = ", ".join(f'"{sui}"' for sui in suis)
        resp_cuis = client.execute(f'GO FROM {suis_str} OVER STY REVERSELY YIELD DISTINCT src(edge) AS cui')
        if resp_cuis.is_empty(): return []
        cuis = [r.values[0].get_sVal().decode("utf-8") for r in resp_cuis.rows()]
        cuis_str = ", ".join(f'"{cui}"' for cui in cuis)
        resp_defs = client.execute(f'GO FROM {cuis_str} OVER DEF YIELD DISTINCT dst(edge) AS def_id')
        if resp_defs.is_empty(): return []
        def_ids = [r.values[0].get_sVal().decode("utf-8") for r in resp_defs.rows()]
        def_ids_str = ", ".join(f'"{d}"' for d in def_ids)
        resp_final = client.execute(f'FETCH PROP ON Definition {def_ids_str} YIELD Definition.DEF')
        if resp_final.is_empty(): return []
        return [r.values[0].get_sVal().decode("utf-8") for r in resp_final.rows()]
    except Exception as e:
        logging.error(f"An error during graph traversal: {e}")
        return []

def rerank_definitions(question, definitions, top_k=15):
    if not definitions: return []
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([[question, d] for d in definitions])
    scored_definitions = sorted(zip(scores, definitions), key=lambda x: x[0], reverse=True)
    top_definitions = [d for _, d in scored_definitions[:top_k]]
    logging.info(f"Re-ranked {len(definitions)} definitions and selected the top {len(top_definitions)}.")
    return top_definitions

def format_shots(shots):
    if not shots: return ""
    examples = []
    for shot in shots:
        inp = shot.get("input", {})
        out = shot.get("Output", {})
        opts = "\\n".join([f"{k}: {v}" for k, v in inp.get("Options", {}).items()])
        example = (
            f"--- Example Start ---\n"
            f"Example Question: {inp.get('Question', '')}\nExample Options:\n{opts}\n"
            f"Example Correct Answer:\n```json\n{json.dumps(out, indent=2)}\n```\n"
            f"--- Example End ---"
        )
        examples.append(example)
    return "\\n\\n".join(examples)

def check_premise_consistency(llm_client, model_name, question, context_str):
    if not context_str: return "NEUTRAL"
    prompt = (
        f"You are a logical validation agent. Determine if the 'Context' supports, contradicts, or is neutral to the 'Question Premise'. "
        f"Answer with ONLY one word: SUPPORTED, CONTRADICTED, or NEUTRAL.\n\n"
        f"Context: {context_str}\nQuestion Premise: {question}\nAnswer:"
    )
    try:
        response = llm_client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        answer = response.choices[0].message.content.strip().upper()
        if "SUPPORTED" in answer: return "SUPPORTED"
        if "CONTRADICTED" in answer: return "CONTRADICTED"
        return "NEUTRAL"
    except Exception as e:
        logging.error(f"Error during premise consistency check: {e}")
        return "NEUTRAL"

def generate_llm_response(llm_client, model_name, question, options, definitions, prompt_assets, consistency_result="", no_rag=False):
    """
    MODIFIED: This function now dynamically builds the prompt based on whether
    RAG is enabled, omitting the Context and Guidance sections in no-RAG mode.
    """
    main_prompt_instruction = prompt_assets.get("prompt", "")
    few_shot_str = format_shots(prompt_assets.get("shots", []))
    options_str = "\\n".join([f"{k}: {v}" for k, v in options.items()])
    output_format_instruction = "You MUST provide your response as a single, valid JSON object."
    
    # --- DYNAMIC PROMPT COMPONENT LOGIC ---
    context_block = ""
    consistency_guidance = ""
    if not no_rag:
        # Only add context and guidance if RAG is enabled.
        context_str = " ".join(definitions) if definitions else "No relevant biomedical context found."
        context_block = f"Context: {context_str}\n\n"
        if consistency_result in ["CONTRADICTED", "NEUTRAL"]:
            consistency_guidance = (
                f"\n--- CRITICAL GUIDANCE ---\nA fact-check determined the context is '{consistency_result}' to the question's premise. This strongly indicates the question is flawed or unanswerable. "
                f"Your primary task is to explain WHY the question is flawed. Set 'cop_index' to the 'None of the above' option if it exists, otherwise set it to -1.\n--- END GUIDANCE ---\n"
            )

    # Assemble the final prompt from the dynamic components
    base_prompt = (
        f"{main_prompt_instruction}\n"
        f"{consistency_guidance}" # Will be empty in no-RAG mode
        f"Examples:\n{few_shot_str}\n\n"
        f"--- CURRENT TASK ---\n"
        f"{context_block}" # Will be empty in no-RAG mode
        f"Question: {question}\nOptions:\n{options_str}\n\n"
        f"Provide your answer. {output_format_instruction}"
    )
    for attempt in range(2):
        prompt = base_prompt + ("\n\nYour previous response was invalid. Please provide ONLY the JSON object." if attempt > 0 else "")
        try:
            response = llm_client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0, response_format={"type": "json_object"})
            raw_text = response.choices[0].message.content
            
            repaired_json_str = repair_json(raw_text)
            
            # 2. Parse the now-guaranteed-to-be-valid JSON string.
            parsed_json = json.loads(repaired_json_str)

            if 'cop_index' not in parsed_json:
                raise ValueError("Output JSON is missing the required 'cop_index' key.")
            
            return parsed_json
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}. Raw response: '{locals().get('raw_text', 'N/A')}'")
            time.sleep(1)
    logging.error(f"Failed to get valid LLM response after multiple attempts.")
    return None