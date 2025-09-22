# RAG_pipeline/generator.py
import logging
import os
import re
import json
from dotenv import load_dotenv
#from openai import OpenAI
from openai import AsyncOpenAI
import spacy
import glob
import asyncio 

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
        self.llm = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("API_KEY"))
        #self.llm = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("API_KEY"))

        # --- NEW: Load SciSpaCy model once for IR RAG mode, handle potential error ---
        self.nlp_model = None

    # --- NEW: All private helper methods for the IR Task workflow ---

    def _load_spacy_model(self):
        """
        Loads the SpaCy model on demand (lazy loading).
        If the model is already loaded, it does nothing.
        """
        # The 'if' check is the key to efficiency: it only loads the model if `self.nlp_model` is None.
        if self.nlp_model:
            return

        logging.info("Loading 'en_ner_bionlp13cg_md' for the first time...")
        try:
            # Assign the loaded model to the instance attribute
            self.nlp_model = spacy.load("en_ner_bionlp13cg_md")
            logging.info("SciSpaCy model loaded successfully.")
        except IOError:
            logging.error(
                "Failed to load SciSpaCy model 'en_ner_bionlp13cg_md'. "
                "It might not be installed correctly. Concept extraction will be disabled."
            )
            # We set it back to None on failure, though this state might be tricky.
            # Depending on desired behavior, you could also raise an error.
            self.nlp_model = None


    async def extract_biomedical_concepts(self, text: str) -> list[str]:
        """
        Extracts concepts from text, ensuring the SpaCy model is loaded first.
        """
        # --- MODIFICATION: Calls the renamed function ---
        self._load_spacy_model()

        # If loading failed, self.nlp_model will be None, and we should exit gracefully.
        if not self.nlp_model: 
            logging.warning("Skipping concept extraction: SciSpaCy model is not available.")
            return []
        
        def _extract():
            doc = self.nlp_model(text)
            return list({ent.text.strip().lower() for ent in doc.ents})
        return await asyncio.to_thread(_extract)


    def _self_correct_query(self, query: str) -> str:
        """
        Takes a raw Cypher query from an LLM and applies a series of 
        programmatic corrections to fix common syntax errors.
        """
        # We start with the original query and build corrections on top of it.
        corrected_query = query
        original_query_for_logging = query
        
        # --- Correction Stage 1: Fix syntax within the WHERE clause ---
        if ' WHERE ' in corrected_query.upper():
            # Isolate the parts of the query for safer replacements
            parts = re.split(r'\bWHERE\b', corrected_query, maxsplit=1, flags=re.IGNORECASE)
            where_clause = parts[1]

            # Correction 1.A: Fix single '=' to '=='
            corrected_where_stage1 = re.sub(r'(?<![=<>!])=(?!=)', r'==', where_clause)
            
            # Correction 1.B: Ensure numeric values after '==' are quoted
            pattern_to_quote = r'(==\s*)(\d+)\b'
            add_quotes = lambda m: f'{m.group(1)}"{m.group(2)}"'
            corrected_where_stage2 = re.sub(pattern_to_quote, add_quotes, corrected_where_stage1)

            # Correction 1.C: Ensure numeric values inside lists are quoted
            def quote_numbers_in_list(match_obj):
                list_str = match_obj.group(0)
                quoted_list_str = re.sub(r'\b(\d+)\b', r'"\1"', list_str)
                return quoted_list_str
            corrected_where_stage3 = re.sub(r'IN\s*\[[^\]]+\]', quote_numbers_in_list, corrected_where_stage2, flags=re.IGNORECASE)
            
            # Rebuild the query with the fully corrected WHERE clause
            corrected_query = parts[0] + 'WHERE' + corrected_where_stage3
            
        # --- Correction Stage 2: Fix incorrect Nebula property access syntax ---
        # This correction is applied to the ENTIRE query string
        match_variable = re.search(r'MATCH\s*\(\s*(\w+)\s*:\s*Paper\s*\)', corrected_query, re.IGNORECASE)
        
        if match_variable:
            variable_name = match_variable.group(1)
            properties = ["pmid", "title", "url", "abstract", "is_paper_exists"] # Added 'is_paper_exists'
            
            for prop in properties:
                incorrect_pattern = f"{variable_name}.{prop}"
                correct_pattern = f"{variable_name}.Paper.{prop}"
                corrected_query = corrected_query.replace(incorrect_pattern, correct_pattern)
        
        # --- Final Logging ---
        if original_query_for_logging != corrected_query:
            logging.info(
                f"Self-correction applied.\n"
                f"  Original: '{original_query_for_logging}'\n"
                f"  Corrected: '{corrected_query}'"
            )
        
        return corrected_query

    # async def _get_cypher_from_llm(self, prompt_assets: dict, question: str, input_key: str, output_key: str) -> str | None:
    #     """
    #     Builds a prompt from the loaded assets and asks the LLM to generate the
    #     Cypher query from scratch.
    #     """
    #     # 1. Get the prompt and format instructions from the loaded assets.
    #     instructions = prompt_assets.get("prompt", "")
    #     format_rules = prompt_assets.get("output_format", "")

    #     if not instructions or not format_rules:
    #         logging.error("Prompt assets are missing 'prompt' or 'output_format' keys.")
    #         return None
        
    #     # 2. Prepare any dynamic variables needed in the prompt.
    #     sanitized_question = question.replace('"', '\\"')
        
    #     concepts = []
    #     concepts_str=""
    #     if "{concepts}" in format_rules: # Check if concepts are needed
    #         concepts = self.extract_biomedical_concepts(question)
    #         concepts_str = json.dumps(concepts)
    #         concepts_str = "- The concepts to match are: "+ concepts_str
    #         format_rules = format_rules.format(concepts=concepts_str, question=sanitized_question)
    #     else:
    #         format_rules = format_rules.format(question=sanitized_question)
    #     # 3. Construct the FINAL prompt to send to the LLM.
    #     #    This is where we combine everything.
    #     #    We pre-fill the concepts/snippet so the LLM knows what values to use.
    #     final_llm_prompt = f"""
    #     {instructions}

    #     {format_rules}
    #     """
        
    #     # 4. Call the LLM with the final combined prompt.
    #     try:
    #         response = await self.llm.chat.completions.create(
    #             model=self.model_name,
    #             messages=[{"role": "user", "content": final_llm_prompt}],
    #             temperature=0.0
    #         )
    #         llm_response = response.choices[0].message.content
            
    #         # 5. Extract the query from the response.
    #         match = re.search(r"```(?:cypher)?\n(.*?)\n```", llm_response, re.DOTALL)
    #         if not match: # A stricter check: if no code block, it failed the instruction.
    #              match = re.search(r'^(MATCH .*)', llm_response, re.DOTALL | re.MULTILINE)
            
    #         if not match:
    #             logging.warning(f"LLM did not return a valid query format. Response: '{llm_response}'")
    #             return None
            
    #         query = match.group(1).strip()
            
    #         # 6. Apply self-correction as a safety net.
    #         #query = self._self_correct_query(query)
            
    #         return query
            
    #     except Exception as e:
    #         logging.error(f"Error getting Cypher from LLM: {e}", exc_info=True)
    #         return None

    def _sanitize_for_cypher(self, text: str) -> str:
        """
        Cleans a string for safe embedding within a Cypher query string literal.
        1. Escapes double quotes.
        2. Replaces newlines and tabs with spaces.
        3. Removes leading/trailing whitespace.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Escape backslashes first, then quotes
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        
        # Replace newline and tab characters with a single space
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Squeeze multiple spaces into one for cleanliness
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    async def _get_execute_cypher_query(self, task_name, prompt_assets: dict, question: str, input_key: str, output_key: str) -> str | None:
        """
        Builds a prompt from the loaded assets and asks the LLM to generate the
        Cypher query from scratch.
        """
        if task_name in ['IR_abstract2pubmedlink']:
            concepts = await self.extract_biomedical_concepts(question)
            #concepts_str = json.dumps(concepts)
            sanitized_question = self._sanitize_for_cypher(question)
        
        # For simple lookups (like PMID), we also sanitize, which is safer
            #sanitized_id = self._sanitize_for_cypher(question)
            cypher_query=f"""MATCH (paper:Paper)-[e:MENTIONS]->(concept:Concept)
            WHERE concept.Concept.name IN {concepts} AND
                paper.Paper.{input_key} CONTAINS "{sanitized_question}"
            RETURN DISTINCT paper.Paper.{output_key}"""
        else:
            cypher_query=f"""MATCH (paper:Paper)
                WHERE paper.Paper.{input_key} == "{question}"
                RETURN DISTINCT paper.Paper.{output_key}"""
        space_name = TASK_TO_SPACE_MAP[task_name]
            
            # This is a synchronous, blocking call, which is acceptable here as it's fast
            # and follows an async LLM call.
        return await self._execute_cypher_and_format(space_name, cypher_query, question, input_key, output_key)

        # # 1. Get the prompt and format instructions from the loaded assets.
        # instructions = prompt_assets.get("prompt", "")
        # format_rules = prompt_assets.get("output_format", "")

        # if not instructions or not format_rules:
        #     logging.error("Prompt assets are missing 'prompt' or 'output_format' keys.")
        #     return None
        
        # # 2. Prepare any dynamic variables needed in the prompt.
        # sanitized_question = question.replace('"', '\\"')
        
        # concepts = []
        # concepts_str=""
        # if "{concepts}" in format_rules: # Check if concepts are needed
        #     concepts = self.extract_biomedical_concepts(question)
        #     concepts_str = json.dumps(concepts)
        #     concepts_str = "- The concepts to match are: "+ concepts_str
        #     format_rules = format_rules.format(concepts=concepts_str, question=sanitized_question)
        # else:
        #     format_rules = format_rules.format(question=sanitized_question)
        # # 3. Construct the FINAL prompt to send to the LLM.
        # #    This is where we combine everything.
        # #    We pre-fill the concepts/snippet so the LLM knows what values to use.
        # final_llm_prompt = f"""
        # {instructions}

        # {format_rules}
        # """
        
        # # 4. Call the LLM with the final combined prompt.
        # try:
        #     response = await self.llm.chat.completions.create(
        #         model=self.model_name,
        #         messages=[{"role": "user", "content": final_llm_prompt}],
        #         temperature=0.0
        #     )
        #     llm_response = response.choices[0].message.content
            
        #     # 5. Extract the query from the response.
        #     match = re.search(r"```(?:cypher)?\n(.*?)\n```", llm_response, re.DOTALL)
        #     if not match: # A stricter check: if no code block, it failed the instruction.
        #          match = re.search(r'^(MATCH .*)', llm_response, re.DOTALL | re.MULTILINE)
            
        #     if not match:
        #         logging.warning(f"LLM did not return a valid query format. Response: '{llm_response}'")
        #         return None
            
        #     query = match.group(1).strip()
            
        #     # 6. Apply self-correction as a safety net.
        #     #query = self._self_correct_query(query)
            
        #     return query
            
        # except Exception as e:
        #     logging.error(f"Error getting Cypher from LLM: {e}", exc_info=True)
        #     return None


    async def _execute_cypher_and_format(self, space_name: str, cypher_query: str, question:str, input_key:str, output_key: str) -> dict:
        """
        Executes a Cypher query using the correct manual try/finally pattern
        for session management.
        """
        if not self.nebula_pool:
            logging.error("Nebula connection pool is not available for Cypher execution.")
            return {output_key: "Unknown"}

        def _db_call():
            session = None
            try:
                session = self.nebula_pool.get_session("root", "nebula")
            # 2. Use the session to execute queries
                session.execute(f"USE {space_name};")
                logging.info(f"Executing in '{space_name}': {cypher_query}")
                result = session.execute(cypher_query)
                
                if result.is_succeeded() and not result.is_empty():
                    value_wrapper=[record.values()[0].as_string() for record in result if record.values()]
                    #value_wrapper = result.rows()[0].values[0]
                    #return {input_key: question, output_key: str(value_wrapper)}
                    return value_wrapper
                else:
                    logging.warning(f"Cypher query failed or returned empty. Error: {result.error_msg() or 'Empty Result'}")
                    #return {input_key: question, output_key: "Unknown"}
                    return ["Unknown"]
            
            except Exception as e:
                # Catch any other exceptions during the process
                logging.error(f"An exception occurred during Cypher execution for '{cypher_query}': {e}", exc_info=True)
                #return {input_key: question, output_key: "Unknown"}
                return ["Unknown"]
                
            finally:
                # 3. CRUCIAL: Always release the session back to the pool
                if session:
                    session.release()

        return await asyncio.to_thread(_db_call)

    async def _handle_ir_task(self, question, options, prompt_assets, task_name, no_rag, input_key=None, output_key=None):
        """NEW: A dedicated handler for all Information Retrieval tasks."""
        
        # --- PATH 1: IR task in RAG (Text-to-Cypher) mode ---
        if not no_rag:
            logging.info(f"Running IR Task '{task_name}' in RAG (Text-to-Cypher) mode.")
            
            # 1. Get Cypher from LLM
            #cypher_query = await self._get_cypher_from_llm(prompt_assets, question, input_key, output_key)
            json_output = await self._get_execute_cypher_query(task_name, prompt_assets, question, input_key, output_key)
            if not json_output:
                output_key = TASK_TO_OUTPUT_KEY_MAP.get(task_name, "error")
                return {output_key: "Unknown"}
            
            # 2. Execute query against NebulaGraph
            space_name = TASK_TO_SPACE_MAP[task_name]
            output_key = TASK_TO_OUTPUT_KEY_MAP[task_name]
            
            # This is a synchronous, blocking call, which is acceptable here as it's fast
            # and follows an async LLM call.
            #return await self._execute_cypher_and_format(space_name, cypher_query, output_key)

            return await generate_llm_response(
                self.llm, self.model_name, question, {}, json_output, 
                prompt_assets, "SUPPORTED", no_rag=no_rag, mode="IR", input_key=input_key, output_key=output_key
            )

        # --- PATH 2: IR task in non-RAG (memory-based) mode ---
        else:
            logging.info(f"Running IR Task '{task_name}' in non-RAG (memory-based) mode.")
            # Use the generic `generate_llm_response` function, but with empty definitions
            # This uses the prompt and shots from the prompt_library.
            return await generate_llm_response(
                self.llm, self.model_name, question, {}, [], 
                prompt_assets, "SUPPORTED", no_rag=no_rag, mode="IR", input_key=input_key, output_key=output_key
            )


    async def _handle_reasoning_task(self, question, options, prompt_assets, task_name, no_rag=False):
        """
        Handles the original RAG and non-RAG pipeline for reasoning tasks.
        """
        final_definitions = []
        consistency_result = "SUPPORTED" 

        if not no_rag:
            if not all([self.st_model, self.faiss_index, self.nebula_pool]):
                raise RuntimeError("RAG components not provided for a RAG-enabled run.")
            
            query = question + " " + " ".join(options.values())
            suis, top_semantic_texts = await retrieve_semantic_nodes(query, self.st_model, self.faiss_index, self.faiss_texts, top_k=30000, top_m=30)
            retrieved_definitions = await get_definitions_from_graph(self.nebula_pool, suis)
            final_definitions = await rerank_definitions(self.cross_encoder, question, retrieved_definitions, top_k=15)
            final_definitions = list(set(top_semantic_texts + final_definitions))
            context_str_for_check = " ".join(final_definitions)
            consistency_result = await check_premise_consistency(self.llm, self.model_name, question, context_str_for_check)
            logging.info(f"Premise consistency check: {consistency_result}")
        else:
            logging.info("Skipping RAG pipeline for reasoning task as per --no-rag flag.")

        return await generate_llm_response(
            self.llm, self.model_name, question, options, final_definitions, 
            prompt_assets, consistency_result, no_rag
        )

    # --- MODIFIED: The main predict function is now a router ---
    async def predict(self, question: str, prompt_assets: dict, task_name: str, no_rag: bool,
                        options: dict = None, input_key: str = None, output_key: str = None):
        """
        Orchestrates the prediction by routing to the correct handler based on task type.
        """
        # --- Route to the correct handler based on task name prefix ---
        if task_name.startswith('IR_'):
            return await self._handle_ir_task(question, options, prompt_assets, task_name, no_rag, input_key, output_key)
        else:
            # --- This is your ORIGINAL, UNCHANGED logic for REASONING tasks ---
            return await self._handle_reasoning_task(question, options, prompt_assets, task_name, no_rag)

