import re
import base64
import random
import os
import base64
import zlib
import pickle
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
from huggingface_hub import snapshot_download

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parsing_utils import extract_test_cases_stdio, extract_python_code
from utils.testing_utils import run_testcase_stdio


def decode(encoded):
    compressed = base64.b64decode(encoded)
    decoded = zlib.decompress(compressed)
    obj = pickle.loads(decoded)
    return obj


def main(args):
    # create output directory
    os.makedirs(f'outputs/evaluation/best_of_n_improvement', exist_ok=True)
    
    # load evaluation set
    if args.benchmark == "taco":
        dataset_name = "dgjun32/UTRL_TACO_EVAL"
    elif args.benchmark == "livecodebench":
        dataset_name = "dgjun32/LiveCodeBench_V2_UTRL_eval"

    repo_path = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        cache_dir=os.path.expanduser('~/.cache/huggingface/datasets'),
    )

    split_name = args.split if args.split else "train"
    split_pattern = f"{repo_path}/data/{split_name}-*.parquet"

    try:
        dataset = load_dataset(
            "parquet",
            data_files=split_pattern,
            split="train",
        )
    except Exception:
        dataset = load_dataset(
            "parquet",
            data_files=f"{repo_path}/data/train-*.parquet",
            split="train",
        )
    
    # get test cases for all problems (or retrieve from cache)
    if args.test_generation_model != "ground_truth":
        tests_for_bon = json.load(open(f'outputs/inference/unit_tests/taco/unit_test_by_{args.test_generation_model.replace("/", "_")}_taco_{args.split}_split.json'))
    elif args.test_generation_model == "ground_truth":
        tests_for_bon = None
        
    print("Evaluating the Solution Best-of-N sampled by Generated Unit Tests")
    evaluation_log = []
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        instance_log = {}
        # get items from the dataset
        gt_test_cases = dataset[i]['gt_test_cases']
        problem_query = dataset[i]['problem_statement']
        multiple_solutions = dataset[i]['sampled_codes'][args.solution_generation_model]

        if args.test_generation_model != "ground_truth":
            test_cases = extract_test_cases_stdio(tests_for_bon[i])
            instance_log['test_cases_for_bon'] = test_cases
        elif args.test_generation_model == "ground_truth":
            test_cases = gt_test_cases
            instance_log['test_cases_for_bon'] = "ground_truth"
        
        
        instance_log['problem_query'] = problem_query
        instance_log['test_cases_for_bon'] = test_cases
        instance_log['candidate_solutions'] = multiple_solutions
        
        # Best-of-N selection generated test cases
        if args.test_generation_model != "ground_truth":
            # filter out the faulty test cases
            print(f"Number of valid test cases: {len(test_cases)}")
            if len(test_cases) > 0:
                # find the best solution measured by self-generated test cases 
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
                best_solution = multiple_solutions[0]
                instance_log['measured_scores'] = [0] * len(multiple_solutions)
                instance_log['best_solution'] = best_solution
                instance_log['best_score'] = "Failed to extract any test cases."
            
            print(f"Estimated Score by Generated Tests: {solution_scores}")
            
            # compute the ground-truth score
            def check_test_case(test_case):
                return run_testcase_stdio(best_solution, test_case)['passed']
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(gt_test_cases))) as executor:
                results = list(executor.map(check_test_case, gt_test_cases))
            n_passed = sum(results)
            
            instance_log['score'] = n_passed / len(gt_test_cases)
            
            if n_passed == len(gt_test_cases):
                instance_log['passed'] = 1.0
            else:
                instance_log['passed'] = 0.0
            
            
        # Best-of-N via ground-truth test cases (upper bound)
        elif args.test_generation_model == "ground_truth":
            # find the best solution measured by ground-truth test cases
            best_solution, best_score = None, 0.0
            
            # Create all (solution, test_case) combinations for parallel execution
            tasks = []
            for sol_idx, solution in enumerate(multiple_solutions):
                solution = extract_python_code(solution)
                for test_case in gt_test_cases:
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
                score = solution_scores[sol_idx] / len(gt_test_cases)
                if score >= best_score:
                    best_solution, best_score = solution, score
            
            instance_log['measured_scores'] = [s / len(gt_test_cases) for s in solution_scores]
            instance_log['best_solution'] = best_solution
            instance_log['best_score'] = best_score
            instance_log['score'] = best_score
            if best_score == 1.0:
                instance_log['passed'] = 1.0
            else:
                instance_log['passed'] = 0.0
        
        evaluation_log.append(instance_log)
        print(f"Solution Passed: {int(instance_log['score'] == 1.0)}")
        print(f"Score by Ground Truth Test Cases: {instance_log['score']}")
        
        # Update progress bar with running metrics
        scores = [log['score'] for log in evaluation_log]
        passed = [log['passed'] for log in evaluation_log]
        avg_score = sum(scores) / len(scores)
        accuracy = sum(passed) / len(passed)
        pbar.set_postfix({
            'avg_score': f'{avg_score:.4f}',
            'accuracy': f'{accuracy:.4f}'
        })
        
        # save the evaluation log
        filename = (
            f"taco_"
            f"test_by_{args.test_generation_model.replace('/', '_')}_"
            f"solution_by_{args.solution_generation_model.replace('/', '_')}_"
            f"best_of_{args.n_samples}_"
            f"taco_{args.split}_split.json"
        )
        filepath = f'outputs/evaluation/best_of_n_improvement/{filename}'
        with open(filepath, 'w') as f:
            json.dump(evaluation_log, f)
        
    # Summarize the evaluation results
    scores = [log['score'] for log in evaluation_log]
    passed = [log['passed'] for log in evaluation_log]
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    accuracy = sum(passed) / len(passed) if passed else 0.0
    
    print("\n" + "="*80)
    print(f"Averaged code score of the chosen solutions: {avg_score:.4f}")
    print(f"Code accuracy of the chosen solutions: {accuracy:.4f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="taco")
    parser.add_argument("--test_generation_model", type=str, default="qwen3_4b")
    parser.add_argument("--solution_generation_model", type=str, default="qwen3_4b")
    parser.add_argument("--best_of_n", action="store_true", required=False)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    
    main(args)
    
         
        
        
    
    




















