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

def retrieve_semantic_nodes(query, model, index, texts, top_k=50, top_m=10):
    """
    MODIFIED: This function now retrieves both the top_k SUIs for graph traversal
    and the top_m full semantic texts for direct use.
    """
    query_vec = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    top_indices = indices[0]
    
    # Get the SUIs for the top_k results (for potential graph traversal)
    top_k_suis = [texts[i]["sui"] for i in top_indices]
    
    # --- YOUR NEW FEATURE ---
    # Get the full text content for the top_m results directly.
    # We use [:top_m] to select the m most similar results from the top_k.
    top_m_texts = [texts[i]["name"] for i in top_indices[:top_m]]
    
    logging.info(f"Retrieved {len(top_k_suis)} SUIs and the top {len(top_m_texts)} semantic texts.")
    return top_k_suis, top_m_texts

def get_definitions_from_graph(pool: ConnectionPool, suis: list):
    if not suis: return []

    session = None
    try:
        # Use a context manager to get a session from the pool.
        # This automatically handles acquiring and releasing the connection.
        with pool.session_context('root', 'nebula') as session:
            # IMPORTANT: You must select the graph space for each new session.
            session.execute("USE petagraph;")

            suis_str = ", ".join(f'"{sui}"' for sui in suis)
            resp_cuis = session.execute(f'GO FROM {suis_str} OVER STY REVERSELY YIELD DISTINCT src(edge) AS cui')
            if resp_cuis.is_empty(): return []
            
            cuis = [r.values[0].get_sVal().decode("utf-8") for r in resp_cuis.rows()]
            cuis_str = ", ".join(f'"{cui}"' for cui in cuis)
            resp_defs = session.execute(f'GO FROM {cuis_str} OVER DEF YIELD DISTINCT dst(edge) AS def_id')
            if resp_defs.is_empty(): return []
            
            def_ids = [r.values[0].get_sVal().decode("utf-8") for r in resp_defs.rows()]
            def_ids_str = ", ".join(f'"{d}"' for d in def_ids)
            resp_final = session.execute(f'FETCH PROP ON Definition {def_ids_str} YIELD Definition.DEF')
            
            if resp_final.is_empty(): return []
            return [r.values[0].get_sVal().decode("utf-8") for r in resp_final.rows()]
    except Exception as e:
        logging.error(f"An error during graph traversal: {e}")
        return []

def rerank_definitions(cross_encoder: CrossEncoder, question, definitions, top_k=15):
    if not definitions: return []
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

def generate_llm_response(
    llm_client, 
    model_name, 
    question, 
    options, 
    definitions, 
    prompt_assets, 
    consistency_result="", 
    no_rag=False,
    mode='full'  # <-- The mode parameter is now included
):
    """
    A unified function that can generate a full JSON response ('full' mode)
    or a fast 'yes'/'no' for probability testing ('forced_choice' mode).
    """
    main_prompt_instruction = prompt_assets.get("prompt", "")
    if options is None: 
        options_str = ""
    else:
        options_str = "\\n".join([f"{k}: {v}" for k, v in options.items()])
    
    context_block = ""
    if not no_rag:
        context_str = " ".join(definitions) if definitions else "No relevant biomedical context found."
        context_block = f"Context: {context_str}\n\n"

    # --- MODE 1: FORCED CHOICE (Lightweight for looping) ---
    if mode == 'forced_choice':
        # This prompt is intentionally simple. It doesn't need few-shots or complex formatting.
        prompt = (
            f"{main_prompt_instruction}\n\n"
            f"--- CURRENT TASK ---\n"
            f"{context_block}"
            f"Question: {question}\nOptions:\n{options_str}\n\n"
            "Based on your expert analysis, is the provided 'correct answer' in the options factually correct? "
            "Respond with ONLY the single word 'yes' or 'no'."
        )
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=3 # We only need a single word
            )
            return response.choices[0].message.content.lower().strip().replace(".", "")
        except Exception as e:
            logging.error(f"API call for forced_choice failed: {e}")
            return "error"

    # --- MODE 2: FULL GENERATION (Your robust version) ---
    elif mode == 'full':
        few_shot_str = format_shots(prompt_assets.get("shots", []))
        consistency_guidance = ""
        if not no_rag and consistency_result in ["CONTRADICTED", "NEUTRAL"]:
            consistency_guidance = (
                f"\n--- CRITICAL GUIDANCE ---\nA fact-check determined the context is '{consistency_result}' to the question's premise. This strongly indicates the question is flawed or unanswerable. "
                f"Your primary task is to explain WHY the question is flawed. Set 'cop_index' to the 'None of the above' option if it exists, otherwise set it to -1.\n--- END GUIDANCE ---\n"
            )

        base_prompt = (
            f"{main_prompt_instruction}\n"
            f"{consistency_guidance}"
            f"Examples:\n{few_shot_str}\n\n"
            f"--- CURRENT TASK ---\n"
            f"{context_block}"
            f"Question: {question}\nOptions:\n{options_str}\n\n"
            f"Provide your answer. {prompt_assets.get('output_format', '')}"
        )
        for attempt in range(2):
            prompt = base_prompt + ("\n\nYour previous response was invalid. Please provide ONLY the JSON object." if attempt > 0 else "")
            try:
                response = llm_client.chat.completions.create(
                    model=model_name, 
                    messages=[{"role": "user", "content": prompt}], 
                    temperature=0.0, 
                    response_format={"type": "json_object"}
                )
                raw_text = response.choices[0].message.content
                parsed_json = json.loads(repair_json(raw_text))
                if 'cop_index' not in parsed_json:
                    raise ValueError("Output JSON is missing the required 'cop_index' key.")
                return parsed_json
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Raw response: '{locals().get('raw_text', 'N/A')}'")
                time.sleep(1)
        
        logging.error(f"Failed to get valid LLM response after multiple attempts.")
        return None
    elif mode=="IR":
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.lower().strip().replace(".", "")
        except Exception as e:
            logging.error(f"API call for forced_choice failed: {e}")
            return "error"
    else:
        logging.error(f"Invalid mode '{mode}' specified for generate_llm_response.")
        return None