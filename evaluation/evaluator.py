import pandas as pd
import glob
import json
from tqdm import tqdm
import os
import logging
import re

class FullDataEval:
    def __init__(self, folder_name, file_pattern="*.json", correct_score=1, incorrect_score=-0.25):
        self.evaluations = []
        self.folder_name = folder_name
        self.correct_score = correct_score
        self.incorrect_score = incorrect_score
        
        # This correctly creates a list of full file paths to be processed.
        self.all_files = glob.glob(os.path.join(self.folder_name, file_pattern))
        logging.info(f"Evaluator initialized. Found {len(self.all_files)} files for pattern '{file_pattern}'.")
        
    def read_json(self, file):
        with open(file, 'r') as json_file:
            return json.load(json_file)

    def calculate_score(self, correct, wrong):
        # This formula correctly calculates the raw penalty score.
        return (correct * self.correct_score) + (wrong * self.incorrect_score)
    
    def evaluate_answer(self, predicted, correct):
        # Stripping whitespace and lowercasing makes the comparison robust.
        return str(predicted).strip().lower() == str(correct).strip().lower()

    def create_dataframe(self, task_name, model_name, correct, wrong, score, prompt_id, kg_rag):
        total = correct + wrong
        df_dict = {
            'model_name': [model_name],
            'task_name': [task_name], 
            'prompt_id': [prompt_id],
            'kg_rag': [kg_rag],
            'total': [total], 
            'correct': [correct], 
            'wrong': [wrong], 
            'score': [score]
        }
        return pd.DataFrame(df_dict)

    def handle_exception(self, task_name, sample_id, model_name, exception):
        logging.error(f"Error processing sample '{sample_id}' in task '{task_name}' with model '{model_name}': {exception}")
        return 1
    
    def evaluate_ir_task(self, task_name, model_name, file_path, prompt_id, kg_rag):
        correct, wrong, exception_count = 0, 0, 0
        
        ir_task_config = {
            "IR_pmid2title":         {"predicted_key": "Title", "correct_key": "Title"},
            "IR_pubmedlink2title":     {"predicted_key": "Title", "correct_key": "Title"},
            "IR_title2pubmedlink":     {"predicted_key": "url", "correct_key": "url"},
            "IR_abstract2pubmedlink": {"predicted_key": "url", "correct_key": "url"}
        }

        config = ir_task_config.get(task_name)
        if not config:
            logging.error(f"No evaluation configuration found for IR task: {task_name}")
            return self.create_dataframe(task_name, model_name, 0, 0, 0, prompt_id, kg_rag)
        
        predicted_key = config["predicted_key"]
        correct_key = config["correct_key"]
        
        all_files_data = self.read_json(file_path)

        for sample in tqdm(all_files_data, desc=f"Evaluating {task_name} with {model_name}"):
            sample_id = sample.get('id', 'unknown_id')
            try:
                predicted_value = sample['gpt_output'][predicted_key]
                correct_value = sample['testbed_data'][correct_key]
                
                if self.evaluate_answer(predicted_value, correct_value):
                    correct += 1
                else:
                    wrong += 1

            except Exception as e:
                exception_count += self.handle_exception(task_name, sample_id, model_name, e)
                # --- FIX: An exception is a wrong answer and should be penalized. ---
                wrong += 1

        score = self.calculate_score(correct, wrong)
        return self.create_dataframe(task_name, model_name, correct, wrong, score, prompt_id, kg_rag)

    def evaluate_reasoning_task(self, task_name, model_name, file_path, prompt_id, kg_rag):
        correct, wrong, exception_count = 0, 0, 0
        all_files_data = self.read_json(file_path)

        for sample in tqdm(all_files_data, desc=f"Evaluating {task_name} with {model_name}"):
            try:
                gpt_output = sample.get('gpt_output', {})
                testbed_data = sample.get('testbed_data', {})

                predicted_index = gpt_output.get('cop_index')
                correct_index = testbed_data.get('correct_index')

                if predicted_index is None:
                    raise KeyError("'cop_index' not found in gpt_output.")
                
                if str(predicted_index) == str(correct_index):
                    correct += 1
                else:
                    wrong += 1

            except Exception as e:
                exception_count += self.handle_exception(task_name, sample.get('id', 'unknown_id'), model_name, e)
                # --- FIX: An exception is a wrong answer and should be penalized. ---
                wrong += 1

        score = self.calculate_score(correct, wrong)
        return self.create_dataframe(task_name, model_name, correct, wrong, score, prompt_id, kg_rag)
    
    def evaluate_reasoning_fake_task(self, task_name, model_name, file_path, prompt_id, kg_rag):
        correct, wrong, exception_count = 0, 0, 0
        all_files_data = self.read_json(file_path)
        SUCCESS_KEYWORDS = [
            'i do not know', 'conceding defeat', 'admit', 'none of the above',
            'acknowled', 'irrelevant', 'fiction', 'all of the above', 
            'nonsensical', 'no correct', 'absurd', 'defy', "i don't know", 
            'defies', 'bizarre', 'illogical', 'cannot answer'
        ]

        for sample in tqdm(all_files_data, desc=f"Evaluating {task_name} with {model_name}"):
            try:
                gpt_output = sample.get('gpt_output', {})
                explanation = str(gpt_output.get('why_correct', '')) + str(gpt_output.get('answer', ''))
                predicted_answer = explanation.lower()

                if any(term in predicted_answer for term in SUCCESS_KEYWORDS):
                    correct += 1
                else:
                    wrong += 1
            
            except Exception as e:
                exception_count += self.handle_exception(task_name, sample.get('id', 'unknown_id'), model_name, e)
                # --- FIX: An exception is a wrong answer and should be penalized. ---
                wrong += 1

        score = self.calculate_score(correct, wrong)
        return self.create_dataframe(task_name, model_name, correct, wrong, score, prompt_id, kg_rag)


    def _finalize_dataframe(self, df):
        """
        --- NEW AND CORRECTED METHOD ---
        This private method takes the raw evaluation DataFrame and prepares it
        for the final report according to your exact specifications.
        """
        if df.empty:
            return df

        # --- FIX 1: Rename the 'score' column to 'penalty_score' ---
        #df.rename(columns={'score': 'penalty_score'}, inplace=True)

        # --- FIX 2: Calculate accuracy. Precision/recall/f1 are removed. ---
        df['accuracy_%'] = (df['correct'] / df['total'] * 100).round(3)
        # df['precision'] = (df['correct'] / (df['correct'] + df['wrong'])).fillna(0)
        # df['recall'] = (df['correct'] / df['total']).fillna(0)
        # df['f1_score'] = (2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])).fillna(0)
        
        # --- FIX 3: Define and apply the exact column order ---
        final_column_order = [
            'model_name',
            'task_name',
            'prompt_id',
            'kg_rag',
            'total',
            'correct',
            'wrong',
            'accuracy_%',
            'score'
        ]
        
        # Filter to only include the desired columns and set their order
        return df[final_column_order]
    
    def run_all_evaluations(self):
        """
        --- REFACTORED AND CORRECTED ---
        This is the main driver method that processes all files, routes them to the
        correct evaluation function, and calculates the final metrics.
        """
        kg_rag_status = 'no' if 'no_rag' in self.folder_name.lower() else 'yes'
        
        for file_path in self.all_files:
            filename = os.path.basename(file_path)
            
            # This regex captures task, prompt_id, and model_name
            pattern = r'((?:reasoning|IR)_.+?)_prompt_(.+?)_model_(.+?)\.json'
            match = re.search(pattern, filename)
            
            if not match:
                logging.warning(f"Could not parse file '{filename}' with pattern. Skipping.")
                continue
            
            task_name, prompt_id, model_name = match.groups()
            
            # Router logic to call the correct evaluation function
            if task_name.startswith('IR_'):
                eval_result = self.evaluate_ir_task(task_name, model_name, file_path, prompt_id, kg_rag_status)
            elif 'reasoning_fake' in task_name:
                eval_result = self.evaluate_reasoning_fake_task(task_name, model_name, file_path, prompt_id, kg_rag_status)
            elif 'reasoning' in task_name:
                eval_result = self.evaluate_reasoning_task(task_name, model_name, file_path, prompt_id, kg_rag_status)
            else:
                logging.warning(f"No evaluator found for task '{task_name}'. Skipping.")
                continue

            self.evaluations.append(eval_result)
        
        if not self.evaluations:
            logging.warning("No evaluation results were generated.")
            return pd.DataFrame()

        # Concatenate all individual results into a single DataFrame
        final_df = pd.concat(self.evaluations, ignore_index=True)
        
        # --- FIX: Call the new method to add the final, crucial calculations ---
        final_df = self._finalize_dataframe(final_df)
        
        return final_df