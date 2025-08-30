# RAG_pipeline/generator.py
import logging
import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from openai import AsyncOpenAI
import spacy

# --- Correctly import all helper functions from the new utils.py file ---
from .utils import (
    retrieve_semantic_nodes,
    get_definitions_from_graph,
    rerank_definitions,
    check_premise_consistency,
    generate_llm_response
)

# --- NEW: Configuration Maps specifically for Information Retrieval Tasks ---
# Maps task names to the correct NebulaGraph space for RAG queries
TASK_TO_SPACE_MAP = {
    'IR_pmid2title': 'medgraph_pmid2title',
    'IR_title2pubmedlink': 'medgraph_title2url',
    'IR_abstract2pubmedlink': 'medgraph2',
    'IR_pubmedlink2title': 'medgraph_url2title'
}

# Maps task names to the expected key in the final JSON output
TASK_TO_OUTPUT_KEY_MAP = {
    'IR_pmid2title': 'Title',
    'IR_title2pubmedlink': 'url',
    'IR_abstract2pubmedlink': 'url',
    'IR_pubmedlink2title': 'Title'
}

class RAGGenerator:
    def __init__(self, model_name, st_model, cross_encoder, faiss_index, faiss_texts, nebula_pool):
        """A lightweight class that RECEIVES shared resources."""
        logging.info(f"Initializing RAG Generator for model: {model_name}...")
        self.model_name = model_name
        
        # Store the shared, pre-loaded resources
        self.st_model = st_model
        self.cross_encoder = cross_encoder 
        self.faiss_index = faiss_index
        self.faiss_texts = faiss_texts
        self.nebula_pool = nebula_pool

        # Initialize the specific component (the LLM client)
        load_dotenv()
        #self.llm = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("API_KEY"))
        self.llm = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("API_KEY"))

        # --- NEW: Load SciSpaCy model once for IR RAG mode, handle potential error ---
        try:
            self.nlp_model = spacy.load("en_ner_bionlp13cg_md")
            logging.info("SciSpaCy model 'en_ner_bionlp13cg_md' loaded successfully.")
        except IOError:
            logging.warning("SciSpaCy model not found. IR RAG mode with concept extraction will not work.")
            self.nlp_model = None

    # --- NEW: All private helper methods for the IR Task workflow ---

    def extract_biomedical_concepts(self, text: str) -> list[str]:
        """
        CORRECTED: A proper class method to extract concepts using the loaded nlp_model.
        """
        if not self.nlp_model:
            logging.warning("Cannot extract concepts because SciSpaCy model is not loaded.")
            return []
        
        doc = self.nlp_model(text)
        # Use a set for automatic deduplication, then convert to list
        return list({ent.text.strip().lower() for ent in doc.ents})

    def build_prompt(self, task_name, concepts, question, input_key="abstract", output_key="url") -> str:
        if task_name in ['IR_abstract2pubmedlink', 'IR_title2pubmedlink']:
            return f"""
            You are a biomedical assistant querying a NebulaGraph knowledge graph that stores scientific papers.

            The graph contains:
            - `Paper` nodes with properties: Paper.abstract, Paper.url, Paper.pmid, Paper.doi
            - `Concept` nodes connected via: (paper:Paper)-[:MENTIONS]->(concept:Concept)

            Your task is to generate a Cypher query that:
            1. Finds all papers mentioning any of the given biomedical concepts.
            2. Filters the papers using the {input_key}.
            3. Returns: DISTINCT paper.Paper.{output_key}

            Use this Cypher format:

            MATCH (paper:Paper)-[e:MENTIONS]->(concept:Concept)
            WHERE concept.Concept.name IN {concepts} AND
                paper.Paper.{input_key} CONTAINS "{question}"
            RETURN DISTINCT paper.Paper.{output_key}

            Only output the Cypher query. Do not explain it.
            """
        else:
            return f"""
            You are a biomedical assistant querying a NebulaGraph knowledge graph that stores scientific papers.

            The graph contains `Paper` nodes with properties: Paper.title, Paper.url, Paper.pmid, Paper.is_paper_exists, etc.
            Your task is to generate a Cypher query that searches the {output_key} property for a given {input_key}.

            Use this Cypher format:

            MATCH (paper:Paper)
            WHERE paper.Paper.{input_key} == "{question}"
            RETURN paper.Paper.{output_key}

            Only output the Cypher query. Do not explain it.
            """



    # def _generate_cypher_prompt(self, question: str) -> str:
    #     """Generates the prompt for the LLM to create a Cypher query."""
    #     concepts = self.extract_biomedical_concepts(question)
    #     input_key=
    #     prompt = self.build_prompt(concepts, question, input_key, output_key)
    #     return prompt
    #     # schema = (
    #     #     "- `Paper` nodes have properties: `pmid`, `title`, `abstract`, `url`.\n"
    #     #     "- When querying, refer to properties like `p.pmid`."
    #     # )
    #     # return (
    #     #     f"You are a NebulaGraph expert. Given the user's question, write a simple Cypher query.\n"
    #     #     f"The graph schema is: {schema}\n"
    #     #     f"Only return the Cypher query inside a markdown code block. Do not explain it.\n"
    #     #     f"Use `CONTAINS` for searching text and `==` for exact IDs. Return only the property the user asks for (e.g., `RETURN p.url`).\n\n"
    #     #     f"User question: \"{question}\""
    #     # )
    
    def _get_cypher_from_llm(self, task_name, question, input_key, output_key) -> str | None:
        """Uses the LLM to translate a natural language question into a Cypher query."""
        concepts = self.extract_biomedical_concepts(question) if task_name in ['IR_abstract2pubmedlink', 'IR_title2pubmedlink'] else []
        prompt = self.build_prompt(task_name, concepts, question, input_key, output_key)
        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            llm_response = response.choices[0].message.content
            
            match = re.search(r"```(cypher)?\n(.*?)```", llm_response, re.DOTALL)
            query = match.group(2).strip() if match else llm_response.strip()

            if "MATCH" not in query:
                logging.error(f"LLM failed to generate valid Cypher. Response: '{llm_response}'")
                return None
            return query
        except Exception as e:
            logging.error(f"Error getting Cypher from LLM: {e}")
            return None

    def _execute_cypher_and_format(self, space_name: str, cypher_query: str, output_key: str) -> dict:
        """
        Executes a Cypher query using the correct manual try/finally pattern
        for session management.
        """
        if not self.nebula_pool:
            logging.error("Nebula connection pool is not available for Cypher execution.")
            return {output_key: "Unknown"}

        session = None  # Initialize session to None
        try:
            # 1. Get a session from the pool
            session = self.nebula_pool.get_session("root", "nebula") # Use your credentials

            # 2. Use the session to execute queries
            session.execute(f"USE {space_name};")
            logging.info(f"Executing in '{space_name}': {cypher_query}")
            result = session.execute(cypher_query)
            
            if result.is_succeeded() and not result.is_empty():
                value_wrapper=[record.values()[0].as_string() for record in result if record.values()][0]
                #value_wrapper = result.rows()[0].values[0]
                return {output_key: str(value_wrapper)}
            else:
                logging.warning(f"Cypher query failed or returned empty. Error: {result.error_msg() or 'Empty Result'}")
                return {output_key: "Unknown"}
        
        except Exception as e:
            # Catch any other exceptions during the process
            logging.error(f"An exception occurred during Cypher execution for '{cypher_query}': {e}", exc_info=True)
            return {output_key: "Unknown"}
            
        finally:
            # 3. CRUCIAL: Always release the session back to the pool
            if session:
                session.release()

    def _handle_ir_task(self, question, options, prompt_assets, task_name, no_rag, input_key=None, output_key=None):
        """NEW: A dedicated handler for all Information Retrieval tasks."""
        
        # --- PATH 1: IR task in RAG (Text-to-Cypher) mode ---
        if not no_rag:
            logging.info(f"Running IR Task '{task_name}' in RAG (Text-to-Cypher) mode.")
            
            # 1. Get Cypher from LLM
            cypher_query = self._get_cypher_from_llm(task_name, question, input_key, output_key)
            if not cypher_query:
                output_key = TASK_TO_OUTPUT_KEY_MAP.get(task_name, "error")
                return {output_key: "Unknown"}
            
            # 2. Execute query against NebulaGraph
            space_name = TASK_TO_SPACE_MAP[task_name]
            output_key = TASK_TO_OUTPUT_KEY_MAP[task_name]
            
            # This is a synchronous, blocking call, which is acceptable here as it's fast
            # and follows an async LLM call.
            return self._execute_cypher_and_format(space_name, cypher_query, output_key)

        # --- PATH 2: IR task in non-RAG (memory-based) mode ---
        else:
            logging.info(f"Running IR Task '{task_name}' in non-RAG (memory-based) mode.")
            # Use the generic `generate_llm_response` function, but with empty definitions
            # This uses the prompt and shots from the prompt_library.
            return generate_llm_response(
                self.llm, self.model_name, question, {}, [], 
                prompt_assets, "SUPPORTED", no_rag=True, mode="IR"
            )

    # --- MODIFIED: The main predict function is now a router ---
    def predict(self, question: str, prompt_assets: dict, task_name: str, no_rag: bool, 
                        options: dict = None, input_key: str = None, output_key: str = None):
        """
        Orchestrates the prediction by routing to the correct handler based on task type.
        """
        # --- Route to the correct handler based on task name prefix ---
        if task_name.startswith('IR_'):
            return self._handle_ir_task(question, options, prompt_assets, task_name, no_rag, input_key, output_key)
        else:
            # --- This is your ORIGINAL, UNCHANGED logic for REASONING tasks ---
            return self._handle_reasoning_task(question, options, prompt_assets, task_name, no_rag)

    # --- UNCHANGED: Your original reasoning logic, moved into its own method ---
    def _handle_reasoning_task(self, question, options, prompt_assets, task_name, no_rag=False):
        """
        Handles the original RAG and non-RAG pipeline for reasoning tasks.
        """
        final_definitions = []
        consistency_result = "SUPPORTED" 

        if not no_rag:
            if not all([self.st_model, self.faiss_index, self.nebula_pool]):
                raise RuntimeError("RAG components not provided for a RAG-enabled run.")
            
            query = question + " " + " ".join(options.values())
            suis, top_semantic_texts =  retrieve_semantic_nodes(query, self.st_model, self.faiss_index, self.faiss_texts, top_k=30000, top_m=30)
            retrieved_definitions =  get_definitions_from_graph(self.nebula_pool, suis)
            final_definitions =  rerank_definitions(self.cross_encoder, question, retrieved_definitions, top_k=15)
            final_definitions = list(set(top_semantic_texts + final_definitions))
            context_str_for_check = " ".join(final_definitions)
            consistency_result =  check_premise_consistency(self.llm, self.model_name, question, context_str_for_check)
            logging.info(f"Premise consistency check: {consistency_result}")
        else:
            logging.info("Skipping RAG pipeline for reasoning task as per --no-rag flag.")

        return  generate_llm_response(
            self.llm, self.model_name, question, options, final_definitions, 
            prompt_assets, consistency_result, no_rag
        )

    # async def predict(self, question, options, prompt_assets, task_name, no_rag=False):
    #     """
    #     Orchestrates the prediction. If no_rag is True, it skips all retrieval.
    #     """
    #     final_definitions = []
    #     consistency_result = "SUPPORTED" # Default for no-RAG or standard tasks

    #     # --- THE CORE NO-RAG LOGIC ---
    #     if not no_rag:
    #         # --- RAG-ENABLED PATH ---
    #         if not all([self.st_model, self.faiss_index, self.nebula_pool]):
    #             raise RuntimeError("RAG components not provided for a RAG-enabled run.")
            
    #         query = question + " " + " ".join(options.values())
    #         #suis = retrieve_semantic_seeds(query, self.st_model, self.faiss_index, self.faiss_texts, top_k=30000)
    #         suis, top_semantic_texts = await retrieve_semantic_nodes(query, self.st_model, self.faiss_index, self.faiss_texts, top_k=30000, top_m=30)
    #         retrieved_definitions = await get_definitions_from_graph(self.nebula_pool, suis)
    #         final_definitions = await rerank_definitions(self.cross_encoder, question, retrieved_definitions, top_k=15)
    #         final_definitions = list(set(top_semantic_texts + final_definitions))
    #         # if 'reasoning_fake' in task_name:
    #         context_str_for_check = " ".join(final_definitions)
    #         consistency_result = await check_premise_consistency(self.llm, self.model_name, question, context_str_for_check)
    #         logging.info(f"Premise consistency check: {consistency_result}")

            
    #     else:
    #         # --- NO-RAG PATH ---
    #         logging.info("Skipping RAG pipeline as per --no-rag flag.")

    #     # Both paths lead to the same final generation step.
    #     # In no-RAG mode, definitions will be empty and consistency will be SUPPORTED.
    #     return await generate_llm_response(
    #         self.llm, self.model_name, question, options, final_definitions, 
    #         prompt_assets, consistency_result, no_rag
    #     )
    
