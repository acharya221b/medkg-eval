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
        
        # --- THE DEFINITIVE FIX IS HERE ---
        # This now correctly creates a list of FULL file paths to be processed.
        self.all_files = glob.glob(os.path.join(self.folder_name, file_pattern))
        logging.info(f"Evaluator initialized. Found {len(self.all_files)} files for pattern '{file_pattern}'.")
        
    def read_json(self, file):
        with open(file, 'r') as json_file:
            return json.load(json_file)

    def calculate_score(self, correct, wrong):
        return (correct * self.correct_score + wrong * self.incorrect_score)

    def create_dataframe(self, task_name, model_name, correct, wrong, score):
        total = correct + wrong
        accuracy = (correct / total * 100) if total > 0 else 0
        df_dict = {
            'model_name': [model_name],
            'task_name': [task_name], 
            'total': [total], 
            'correct': [correct], 
            'wrong': [wrong], 
            'accuracy_%': [accuracy],
            'score': [score]
        }
        return pd.DataFrame(df_dict)

    def handle_exception(self, task_name, sample_id, model_name, exception):
        logging.error(f"Error processing sample '{sample_id}' in task '{task_name}' with model '{model_name}': {exception}")
        return 1

    def evaluate_reasoning_task(self, task_name, model_name, file_path):
        """A single evaluation method for all our reasoning tasks."""
        correct, wrong, exception_count = 0, 0, 0
        all_files_data = self.read_json(file_path)

        for sample in tqdm(all_files_data, desc=f"Evaluating {task_name}"):
            try:
                # The 'testbed_data' column was not being correctly created. This is now fixed in main.py.
                gpt_output = sample.get('gpt_output', {})
                testbed_data = sample.get('testbed_data', {})

                if 'cop_index' not in gpt_output:
                    raise KeyError("'cop_index' not found in gpt_output.")
                                
                predicted_index = gpt_output.get('cop_index')
                correct_index = testbed_data.get('correct_index')

                if predicted_index is None:
                    raise KeyError("'cop_index' not found in gpt_output.")
                
                # Compare predicted index with correct index
                if str(predicted_index) == str(correct_index):
                    correct += 1
                else:
                    wrong += 1

            except Exception as e:
                exception_count += self.handle_exception(task_name, sample.get('id', 'unknown_id'), model_name, e)
                wrong += 1

        logging.info(f"Results for {task_name} and {model_name}: Correct={correct}, Wrong={wrong}, Exceptions={exception_count}")
        score = self.calculate_score(correct, wrong)
        return self.create_dataframe(task_name, model_name, correct, wrong, score)
    
    def evaluate_reasoning_fake_task(self, task_name, model_name, file_path):
        """
        A special evaluation method for the 'reasoning_fake' task where success
        is defined by the LLM recognizing the question is nonsensical.
        """
        correct, wrong, exception_count = 0, 0, 0
        all_files_data = self.read_json(file_path)

        # The keywords that indicate a successful recognition of a fake question.
        SUCCESS_KEYWORDS = [
            'i do not know', 'conceding defeat', 'admit', 'none of the above',
            'acknowled', 'irrelevant', 'fiction', 'all of the above', 
            'nonsensical', 'no correct', 'absurd', 'defy', "i don't know", 
            'defies', 'bizarre', 'illogical', 'cannot answer'
        ]

        for sample in tqdm(all_files_data, desc=f"Evaluating {task_name} with {model_name}"):
            try:
                gpt_output = sample.get('gpt_output', {})
                
                # For this task, we look at the text explanation, not an index.
                # We'll check multiple likely keys for the explanation text.
                explanation = str(gpt_output.get('why_correct', '')) + str(gpt_output.get('answer', ''))
                predicted_answer = explanation.lower()

                # Check if any of the success keywords are in the LLM's response.
                if any(term in predicted_answer for term in SUCCESS_KEYWORDS):
                    correct += 1
                else:
                    wrong += 1
            
            except Exception as e:
                exception_count += self.handle_exception(task_name, sample.get('id', 'unknown_id'), model_name, e)
                wrong += 1

        logging.info(f"Results for {task_name} and {model_name}: Correct={correct}, Wrong={wrong}, Exceptions={exception_count}")
        score = self.calculate_score(correct, wrong)
        return self.create_dataframe(task_name, model_name, correct, wrong, score)

    def run_all_evaluations(self):
        for file_path in self.all_files:
            filename = os.path.basename(file_path)
            
            # --- MODIFICATION: Parse task and model name from filename ---
            match = re.search(r'(reasoning_\w+)_prompt_.*?_model_(.*?)\.json', filename)
            if not match:
                logging.warning(f"Could not parse task/model name from '{filename}'. Skipping.")
                continue
            
            task_name, model_name = match.groups()
            
            if 'reasoning_fake' in task_name:
                eval_result = self.evaluate_reasoning_fake_task(task_name, model_name, file_path)
            else:
                eval_result = self.evaluate_reasoning_task(task_name, model_name, file_path)
            self.evaluations.append(eval_result)
        
        return pd.concat(self.evaluations, ignore_index=True) if self.evaluations else pd.DataFrame()