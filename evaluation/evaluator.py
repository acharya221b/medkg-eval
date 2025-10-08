# evaluation/evaluator.py

import pandas as pd
import glob
import json
from tqdm import tqdm
import os
import logging
import re
import numpy as np

class FullDataEval:
    """
    A comprehensive evaluation class that processes JSON results for both
    reasoning and information retrieval tasks, now with support for subset-based
    statistical analysis.
    """
    def __init__(self, folder_name, all_files, correct_score=1, incorrect_score=-0.25):
        self.folder_name = folder_name
        self.correct_score = correct_score
        self.incorrect_score = incorrect_score
        self.all_files = all_files
        logging.info(f"Evaluator initialized with {len(self.all_files)} specific files to process.")


    def read_json(self, file_path):
        """Safely reads and loads a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Could not read or parse JSON file: {file_path}. Error: {e}")
            return [] # Return an empty list to prevent crashes

    def calculate_score(self, correct, wrong):
        """Calculates the raw penalty score."""
        return (correct * self.correct_score) + (wrong * self.incorrect_score)
    
    def evaluate_answer(self, predicted, correct):
        """Performs a robust, case-insensitive string comparison."""
        return str(predicted).strip().lower() == str(correct).strip().lower()

    def handle_exception(self, task_name, sample_id, model_name, exception):
        """Logs exceptions encountered during sample evaluation."""
        logging.error(f"Error processing sample '{sample_id}' in '{task_name}' with model '{model_name}': {exception}")
        return 1 # Returns 1 to count as a "wrong" answer

    def _get_eval_function_for_task(self, task_name):
        """A router to select the correct evaluation logic for a single sample."""
        if task_name.startswith('IR_'):
            return self._get_ir_eval_function(task_name)
        elif 'reasoning_fake' in task_name:
            return self._evaluate_reasoning_fake_sample
        elif 'reasoning' in task_name:
            return self._evaluate_reasoning_sample
        return None
    
    def _evaluate_reasoning_sample(self, sample):
        """Evaluates a single sample for a standard reasoning task."""
        try:
            predicted = sample.get('gpt_output', {}).get('cop_index')
            correct = sample.get('testbed_data', {}).get('correct_index')
            if predicted is None or correct is None: return False
            return str(predicted) == str(correct)
        except Exception:
            return False

    def _evaluate_reasoning_fake_sample(self, sample):
        """Evaluates a single sample for the 'fake' reasoning task."""
        SUCCESS_KEYWORDS = ['i do not know', 'conceding defeat', 'admit', 'none of the above', 'acknowled', 'irrelevant', 'fiction', 'all of the above', 'nonsensical', 'no correct', 'absurd', 'defy', "i don't know", 'defies', 'bizarre', 'illogical', 'cannot answer']
        try:
            explanation = str(sample.get('gpt_output', {}).get('why_correct', '')) + str(sample.get('gpt_output', {}).get('answer', ''))
            return any(term in explanation.lower() for term in SUCCESS_KEYWORDS)
        except Exception:
            return False
    
    def _get_ir_eval_function(self, task_name):
        """Returns a specialized evaluation function for a given IR task."""
        ir_task_config = {
            "IR_pmid2title":          {"predicted_key": "Title", "correct_key": "Title"},
            "IR_pubmedlink2title":      {"predicted_key": "Title", "correct_key": "Title"},
            "IR_title2pubmedlink":      {"predicted_key": "url", "correct_key": "url"},
            "IR_abstract2pubmedlink": {"predicted_key": "url", "correct_key": "url"}
        }
        config = ir_task_config.get(task_name)
        if not config: return None

        def _eval_func(sample):
            try:
                predicted = sample['gpt_output'][config["predicted_key"]]
                correct = sample['testbed_data'][config["correct_key"]]
                return self.evaluate_answer(predicted, correct)
            except (KeyError, TypeError): # Catch errors from missing keys or non-dict objects
                return False
        return _eval_func

    def _evaluate_single_file(self, file_path, task_name, model_name, subset_size):
        """
        Core evaluation logic for a single file, now operating on subsets.
        Returns a dictionary with overall results and a list of subset accuracies.
        """
        all_data = self.read_json(file_path)
        eval_function = self._get_eval_function_for_task(task_name)
        if not eval_function or not all_data:
            return None

        total_correct = 0
        total_wrong = 0
        exception_count = 0
        subset_accuracies = []
        
        for i in range(0, len(all_data), subset_size):
            subset = all_data[i : i + subset_size]
            subset_correct = 0
            
            for sample in subset:
                try:
                    if eval_function(sample):
                        subset_correct += 1
                except Exception as e:
                    # This exception is for unexpected errors in the eval logic itself
                    exception_count += self.handle_exception(task_name, sample.get('id'), model_name, e)
            
            subset_wrong = len(subset) - subset_correct
            if len(subset) > 0:
                accuracy = (subset_correct / len(subset)) * 100
                subset_accuracies.append(accuracy)
            
            total_correct += subset_correct
            total_wrong += subset_wrong

        logging.info(f"Results for {task_name}/{model_name}: Correct={total_correct}, Wrong={total_wrong}, Exceptions={exception_count}")
        return {
            "total_correct": total_correct,
            "total_wrong": total_wrong,
            "subset_accuracies": subset_accuracies
        }

    def run_all_evaluations(self, subset_size=100):
        """
        Main driver method. Processes all files, generates reports, and returns
        both a summary DataFrame and a dictionary with detailed subset accuracies.
        """
        main_report_data = []
        subset_details_data = {}

        for file_path in self.all_files:
            filename = os.path.basename(file_path)
            pattern = r'((?:reasoning|IR)_.+?)_prompt_(.+?)_model_(.+?)\.json'
            match = re.search(pattern, filename)
            
            if not match:
                logging.warning(f"Could not parse filename '{filename}' with pattern. Skipping.")
                continue
            
            task_name, prompt_id, model_name = match.groups()
            kg_rag_status = 'no' if 'no_rag' in self.folder_name.lower() else 'yes'
            run_key = f"{task_name}_{model_name}_{prompt_id}_{kg_rag_status}"

            eval_results = self._evaluate_single_file(file_path, task_name, model_name, subset_size)
            
            if eval_results:
                total_correct = eval_results["total_correct"]
                total_wrong = eval_results["total_wrong"]
                total = total_correct + total_wrong
                overall_accuracy = (total_correct / total * 100) if total > 0 else 0
                
                accuracies = eval_results["subset_accuracies"]
                
                # --- NEW STATISTICAL CALCULATIONS ---
                avg_subset_accuracy = np.mean(accuracies) if accuracies else 0.0
                std_dev_subset_accuracy = np.std(accuracies) if accuracies else 0.0
                
                # --- SAVE SUBSET DATA ---
                subset_details_data[run_key] = accuracies
                
                report_row = {
                    'model_name': model_name,
                    'task_name': task_name,
                    'prompt_id': prompt_id,
                    'kg_rag': kg_rag_status,
                    'total_samples': total,
                    'correct': total_correct,
                    'wrong': total_wrong,
                    'overall_accuracy_%': f"{overall_accuracy:.2f}",
                    'avg_subset_accuracy_%': f"{avg_subset_accuracy:.2f}",
                    'std_dev_subset_accuracy': f"{std_dev_subset_accuracy:.2f}",
                    'penalty_score': self.calculate_score(total_correct, total_wrong)
                }
                main_report_data.append(report_row)
        
        main_report_df = pd.DataFrame(main_report_data)
        return main_report_df, subset_details_data