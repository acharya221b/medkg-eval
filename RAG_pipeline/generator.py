# RAG_pipeline/generator.py
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

# --- Correctly import all helper functions from the new utils.py file ---
from .utils import (
    retrieve_semantic_seeds,
    retrieve_semantic_nodes,
    get_definitions_from_graph,
    rerank_definitions,
    check_premise_consistency,
    generate_llm_response
)

class RAGGenerator:
    def __init__(self, model_name, st_model, faiss_index, faiss_texts, nebula_client):
        """A lightweight class that RECEIVES shared resources."""
        logging.info(f"Initializing RAG Generator for model: {model_name}...")
        self.model_name = model_name
        
        # Store the shared, pre-loaded resources
        self.st_model = st_model
        self.faiss_index = faiss_index
        self.faiss_texts = faiss_texts
        self.nebula_client = nebula_client

        # Initialize the specific component (the LLM client)
        load_dotenv()
        self.llm = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("API_KEY"))

    def predict(self, question, options, prompt_assets, task_name, no_rag=False):
        """
        Orchestrates the prediction. If no_rag is True, it skips all retrieval.
        """
        final_definitions = []
        consistency_result = "SUPPORTED" # Default for no-RAG or standard tasks

        # --- THE CORE NO-RAG LOGIC ---
        if not no_rag:
            # --- RAG-ENABLED PATH ---
            if not all([self.st_model, self.faiss_index, self.nebula_client]):
                raise RuntimeError("RAG components not provided for a RAG-enabled run.")
            
            query = question + " " + " ".join(options.values())
            #suis = retrieve_semantic_seeds(query, self.st_model, self.faiss_index, self.faiss_texts, top_k=30000)
            suis, top_semantic_texts = retrieve_semantic_nodes(query, self.st_model, self.faiss_index, self.faiss_texts, top_k=30000, top_m=30)
            retrieved_definitions = get_definitions_from_graph(self.nebula_client, suis)
            final_definitions = rerank_definitions(question, retrieved_definitions, top_k=15)
            final_definitions = list(set(top_semantic_texts + final_definitions))
            # if 'reasoning_fake' in task_name:
            context_str_for_check = " ".join(final_definitions)
            consistency_result = check_premise_consistency(self.llm, self.model_name, question, context_str_for_check)
            logging.info(f"Premise consistency check: {consistency_result}")

            
        else:
            # --- NO-RAG PATH ---
            logging.info("Skipping RAG pipeline as per --no-rag flag.")

        # Both paths lead to the same final generation step.
        # In no-RAG mode, definitions will be empty and consistency will be SUPPORTED.
        return generate_llm_response(
            self.llm, self.model_name, question, options, final_definitions, 
            prompt_assets, consistency_result, no_rag
        )