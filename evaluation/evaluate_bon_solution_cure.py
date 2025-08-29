import re
import base64
import zlib
import pickle
import random
import os
import ast 
import astor  # astor is used to convert AST nodes back to source code
import json
import math
import tempfile
import subprocess
import time
import concurrent
import argparse
from typing import List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set cache directory to avoid permission issues
os.environ['HF_DATASETS_CACHE'] = os.path.expanduser('~/.cache/huggingface/datasets')

import tqdm
from tqdm import tqdm
import torch
import datasets
import transformers
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset


from utils.parsing_utils import extract_test_cases_cure, extract_python_code
from utils.testing_utils import run_testcase_stdio

def decode(encoded):
    compressed = base64.b64decode(encoded)
    decoded = zlib.decompress(compressed)
    obj = pickle.loads(decoded)
    return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_generation_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--solution_generation_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--best_of_n", action="store_true", required=False)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
                  
    dataset = load_dataset(
        "BAAI/TACO",
        cache_dir=os.path.expanduser('~/.cache/huggingface/datasets')
    )[args.split]
    problem_key = "question"
    gt_tests = []
    for i in range(len(dataset)):
        print(i)
        gt_test = []
        try:
            test_cases = eval(dataset[i]['input_output'])
        except:
            test_cases = {'inputs': [], 'outputs': []}
        for inp, out in zip(test_cases['inputs'], test_cases['outputs']):
            gt_test.append({'input': inp, 'output': out})
        gt_tests.append(gt_test)
            
    
    # get multiple solutions per problem
    solutions = json.load(
        open(f'results/inference/taco_solutions/solution_by_{args.solution_generation_model.replace("/", "_")}_best_of_{args.n_samples}_taco_{args.split}_split.json'))
    
    # get test cases for best-of-N selection 
    if args.test_generation_model != "ground_truth":
        tests_for_bon = json.load(open(f'results/inference/taco_unit_tests/unit_test_by_{args.test_generation_model.replace("/", "_")}_best_of_16_taco_{args.split}_split.json'))
    elif args.test_generation_model == "ground_truth":
        tests_for_bon = gt_tests

    print("Evaluating the Solution Best-of-N sampled by Generated Unit Tests")
    evaluation_log = []
    for i in tqdm(range(len(dataset))):
        try:
            instance_log = {}
            problem_query = dataset[i][problem_key]
            instance_log['problem_query'] = problem_query
            if args.test_generation_model != "ground_truth":
                test_cases = extract_test_cases_cure(tests_for_bon[i])
                instance_log['test_cases_for_bon'] = test_cases
            multiple_solutions: List[str] = solutions[i]
            instance_log['candidate_solutions'] = multiple_solutions
            
            if args.test_generation_model != "ground_truth":
                # filter out the faulty test cases
                print(f"Number of valid test cases: {len(test_cases)}")
                if len(test_cases) > 0:
                    # find the best solution measured by generated test cases 
                    best_solution, best_score = None, 0.0
                    # Create all (solution, test_case) combinations for parallel execution
                    tasks = []
                    for sol_idx, solution in enumerate(multiple_solutions):
                        solution = extract_python_code(solution)
                        for test_case in test_cases:
                            tasks.append((sol_idx, solution, test_case))
                    # Run all test cases in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                        future_to_task = {
                            executor.submit(run_testcase_stdio, task[1], task[2]): task 
                            for task in tasks
                        }
                        # Collect results
                        solution_scores = [0] * len(multiple_solutions)
                        for future in concurrent.futures.as_completed(future_to_task):
                            sol_idx, solution, test_case = future_to_task[future]
                            try:
                                result = future.result()
                                
                                if result['passed']:
                                    solution_scores[sol_idx] += 1
                            except Exception as exc:
                                print(f'Test case generated an exception: {exc}')
                    # Find the best solution
                    for sol_idx, solution in enumerate(multiple_solutions):
                        solution = extract_python_code(solution)
                        score = solution_scores[sol_idx] / len(test_cases)
                        if score >= best_score:
                            best_solution, best_score = solution, score
                    instance_log['measured_scores'] = [s / len(test_cases) for s in solution_scores]
                    instance_log['best_solution'] = best_solution
                    instance_log['best_score'] = best_score
                else:
                    instance_log['measured_scores'] = [0] * len(multiple_solutions)
                    instance_log['best_solution'] = multiple_solutions[0]
                    instance_log['best_score'] = "Failed to extract any test cases."
                print(f"Estimated Score by Generated Tests: {solution_scores}")
                
                # compute the ground-truth score
                def check_test_case(test_case):
                    return run_testcase_stdio(best_solution, test_case)['passed']
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(gt_tests[i]))) as executor:
                    results = list(executor.map(check_test_case, gt_tests[i]))
                n_passed = sum(results)
                
                instance_log['score'] = n_passed / len(gt_tests[i])
                
                if n_passed == len(gt_tests[i]):
                    instance_log['passed'] = 1.0
                else:
                    instance_log['passed'] = 0.0
                
                
            # Best-of-N selection via ground-truth test cases
            elif args.test_generation_model == "ground_truth":
                # find the best solution measured by ground-truth test cases
                best_solution, best_score = None, 0.0
                
                # Create all (solution, test_case) combinations for parallel execution
                tasks = []
                for sol_idx, solution in enumerate(multiple_solutions):
                    solution = extract_python_code(solution)
                    for test_case in gt_tests[i]:
                        tasks.append((sol_idx, solution, test_case))
                
                # Run all test cases in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                    future_to_task = {
                        executor.submit(run_testcase_stdio, task[1], task[2]): task 
                        for task in tasks
                    }
                    
                    # Collect results
                    solution_scores = [0] * len(multiple_solutions)
                    for future in concurrent.futures.as_completed(future_to_task):
                        sol_idx, solution, test_case = future_to_task[future]
                        try:
                            result = future.result()
                            if result['passed']:
                                solution_scores[sol_idx] += 1
                        except Exception as exc:
                            print(f'Test case generated an exception: {exc}')
                
                # Find the best solution
                for sol_idx, solution in enumerate(multiple_solutions):
                    solution = extract_python_code(solution)
                    score = solution_scores[sol_idx] / len(gt_tests[i])
                    if score >= best_score:
                        best_solution, best_score = solution, score
                instance_log['measured_scores'] = [s / len(gt_tests[i]) for s in solution_scores]
                instance_log['chosen_solution'] = best_solution
                instance_log['best_score'] = best_score
                instance_log['score'] = best_score
                if best_score == 1.0:
                    instance_log['passed'] = 1.0
                else:
                    instance_log['passed'] = 0.0
            
            evaluation_log.append(instance_log)
            print(f"Solution Passed: {int(instance_log['score'] == 1.0)}")
            print(f"Score by Ground Truth Test Cases: {instance_log['score']}")
            
            # save the evaluation log
            filename = (
                f"taco_"
                f"test_by_{args.test_generation_model.replace('/', '_')}_"
                f"solution_by_{args.solution_generation_model.replace('/', '_')}_"
                f"best_of_{args.n_samples}_"
                f"taco_{args.split}_split.json"
            )
            filepath = f'results/evaluation/{filename}'
            with open(filepath, 'w') as f:
                json.dump(evaluation_log, f)
        except Exception as e:
            print(f'Error message: {e}')
        
        
        
    
    




















