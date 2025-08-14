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
#from openai import OpenAI
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
import asyncio 

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

# def connect_nebula():
#     """
#     Initializes and returns the NebulaGraph connection pool.
#     """
#     try:
#         config = Config()
#         config.max_connection_pool_size = 10 # Or whatever size you need
#         pool = ConnectionPool()
#         pool.init([("127.0.0.1", 9669)], config)
        
#         # Verify the pool is alive but DON'T check out a session here.
#         logging.info("Successfully initialized NebulaGraph connection pool.")
#         return pool
#     except Exception as e:
#         logging.error(f"Failed to initialize NebulaGraph connection pool: {e}")
#         return None

def load_faiss_index():
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(TEXTS_FILE, "r") as f: texts = json.load(f)
        logging.info(f"FAISS index loaded with {index.ntotal} vectors.")
        return index, texts
    except Exception as e:
        logging.error(f"Could not load FAISS index or texts file: {e}")
        return None, None

async def retrieve_semantic_nodes(query, model, index, texts, top_k=50, top_m=10):
    """
    MODIFIED: This function now retrieves both the top_k SUIs for graph traversal
    and the top_m full semantic texts for direct use.
    """
    # Sentence encoding is CPU-bound and blocks the event loop.
    # We run it in a separate thread to keep the pipeline responsive.
    def _blocking_encode_and_search():
        query_vec = model.encode([query], convert_to_numpy=True)
        _, indices = index.search(query_vec, top_k)
        return indices[0]

    top_indices = await asyncio.to_thread(_blocking_encode_and_search)
    
    # Get the SUIs for the top_k results (for potential graph traversal)
    top_k_suis = [texts[i]["sui"] for i in top_indices]
    
    # --- YOUR NEW FEATURE ---
    # Get the full text content for the top_m results directly.
    # We use [:top_m] to select the m most similar results from the top_k.
    top_m_texts = [texts[i]["name"] for i in top_indices[:top_m]]
    
    logging.info(f"Retrieved {len(top_k_suis)} SUIs and the top {len(top_m_texts)} semantic texts.")
    return top_k_suis, top_m_texts

async def get_definitions_from_graph(pool: ConnectionPool, suis: list):
    if not suis: return []

    # The nebula client's execute method is blocking I/O.
    # We run it in a thread to prevent it from stalling other async tasks.
    def _blocking_graph_queries():
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

    return await asyncio.to_thread(_blocking_graph_queries)

async def rerank_definitions(cross_encoder: CrossEncoder, question, definitions, top_k=15):
    if not definitions: return []
    # Cross-encoder prediction is also a heavy, CPU-bound task.
    def _blocking_rerank():
        #cross_encoder = CrossEncoder('pritamdeka/S-PubMedBert-MS-MARCO')
        scores = cross_encoder.predict([[question, d] for d in definitions])
        scored_definitions = sorted(zip(scores, definitions), key=lambda x: x[0], reverse=True)
        return [d for _, d in scored_definitions[:top_k]]

    top_definitions = await asyncio.to_thread(_blocking_rerank)
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

async def check_premise_consistency(llm_client, model_name, question, context_str):
    if not context_str: return "NEUTRAL"
    prompt = (
        f"You are a logical validation agent. Determine if the 'Context' supports, contradicts, or is neutral to the 'Question Premise'. "
        f"Answer with ONLY one word: SUPPORTED, CONTRADICTED, or NEUTRAL.\n\n"
        f"Context: {context_str}\nQuestion Premise: {question}\nAnswer:"
    )
    try:
        response = await llm_client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        answer = response.choices[0].message.content.strip().upper()
        if "SUPPORTED" in answer: return "SUPPORTED"
        if "CONTRADICTED" in answer: return "CONTRADICTED"
        return "NEUTRAL"
    except Exception as e:
        logging.error(f"Error during premise consistency check: {e}")
        return "NEUTRAL"

async def generate_llm_response(llm_client, model_name, question, options, definitions, prompt_assets, consistency_result="", no_rag=False):
    """
    MODIFIED: This function now dynamically builds the prompt based on whether
    RAG is enabled, omitting the Context and Guidance sections in no-RAG mode.
    """
    main_prompt_instruction = prompt_assets.get("prompt", "")
    few_shot_str = format_shots(prompt_assets.get("shots", []))
    options_str = "\\n".join([f"{k}: {v}" for k, v in options.items()])
    # output_format_instruction = (
    # "You MUST provide your response as a single, valid JSON object with the following keys:\n"
    # "1. `cop_index`: The integer index of the correct option.\n"
    # "2. `answer`: The full string value of the correct option.\n"
    # "3. `why_correct`: This MUST be a list of strings. Each string in the list is a distinct, logical step in your reasoning. You must follow this exact 4-step structure for the list:\n"
    # "   - Step 1 (string 1): Quote the exact sentences or phrases from the provided Context that are most relevant to the Question. If there is no Context available, just state 'I don't know'.\n"
    # "   - Step 2 (string 2): Analyze and state the key information provided in the Question itself (e.g., specific conditions, patient details).\n"
    # "   - Step 3 (string 3): If Context is available, explain how the Context supports the key information in the Question, otherwise state 'I don't know'.\n"
    # "   - Step 4 (string 4): Explicitly state why the chosen option is correct based on your reasoning.\n"
    # "4. `why_others_incorrect`: A brief explanation for why each of the other options is wrong."
    # )
    # output_format_instruction = (
    #     "You MUST provide your response as a single, valid JSON object with the following keys:\n"
    #     "1. `cop_index`: The integer index of the correct option.\n"
    #     "2. `answer`: The full string value of the correct option.\n"
    #     "3. `why_correct`: A detailed explanation of only the correct answer. This explanation MUST follow a specific three-part structure:\n"
    #     "   - First, briefly state the key concepts in the question.\n"
    #     "   - Second, quote all the exact sentences from the Context that directly support your answer.\n"
    #     "   - Finally, provide a concluding sentence that links the evidence to the chosen answer.\n"
    #     "4. `why_others_incorrect`: A brief explanation for why each of the other options is wrong."
    # )
    # output_format_instruction = ("You MUST provide your response as a single, valid JSON object with the following keys.\n"                           
    #                             f"{prompt_assets.get('output_format', '')}"
    #                              )
    
    # # --- DYNAMIC PROMPT COMPONENT LOGIC ---
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
        f"Provide your answer. {prompt_assets.get('output_format', '')}"
    )
    for attempt in range(2):
        prompt = base_prompt + ("\n\nYour previous response was invalid. Please provide ONLY the JSON object." if attempt > 0 else "")
        try:
            #print(prompt)
            response = await llm_client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0, response_format={"type": "json_object"})
            raw_text = response.choices[0].message.content
            # response = llm_client.complete(prompt)
            # raw_text = response.text
            repaired_json_str = repair_json(raw_text)
            
            # 2. Parse the now-guaranteed-to-be-valid JSON string.
            parsed_json = json.loads(repaired_json_str)

            if 'cop_index' not in parsed_json:
                raise ValueError("Output JSON is missing the required 'cop_index' key.")
            #print(parsed_json)
            return parsed_json
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}. Raw response: '{locals().get('raw_text', 'N/A')}'")
            await asyncio.sleep(1)
    logging.error(f"Failed to get valid LLM response after multiple attempts.")
    return None