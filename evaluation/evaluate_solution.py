import re
import base64
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

import tqdm
from tqdm import tqdm
import torch
import datasets
import transformers
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parsing_utils import extract_test_cases_stdio, extract_python_code
from utils.testing_utils import run_testcase_stdio

# Set cache directory to avoid permission issues
os.environ['HF_DATASETS_CACHE'] = os.path.expanduser('~/.cache/huggingface/datasets')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_generation_model", type=str, default="Qwen/Qwen3-4B")
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
    
    # get single solution per problem (or retrieve from cache)
    solutions = json.load(open(f'results/inference/taco_solutions/solution_by_{args.solution_generation_model.replace("/", "_")}_taco_{args.split}_split.json'))
    evaluation_log = []
    for i in tqdm(range(len(dataset)), desc="Evaluating the Solution using Ground-truth Unit test"):
        try:
            instance_log = {}
            problem_query = dataset[i][problem_key]
            instance_log['global_idx'] = i
            instance_log['problem_query'] = problem_query
            solution: List[str] = extract_python_code(solutions[i])
            instance_log['solution'] = solution
            instance_log['source'] = dataset[i]['source']
            
            # Evaluate solution with gt tests 
            def check_test_case(test_case):
                return run_testcase_stdio(solution, test_case)['passed']
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(gt_tests[i]))) as executor:
                results = list(executor.map(check_test_case, gt_tests[i]))
            n_passed = sum(results)
            
            instance_log['gt_score'] = n_passed / len(gt_tests[i])
            
            if n_passed == len(gt_tests[i]):
                instance_log['passed'] = 1.0
            else:
                instance_log['passed'] = 0.0
                
            evaluation_log.append(instance_log)
            print(f"Solution Passed: {int(instance_log['gt_score'] == 1.0)}")
            print(f"Score by Ground Truth Test Cases: {instance_log['gt_score']}")
            
            # save the evaluation log
            filename = (
                f"solution_by_{args.solution_generation_model.replace('/', '_')}_"
                f"taco_{args.split}_split.json"
            )
            filepath = f'results/evaluation/{filename}'
            with open(filepath, 'w') as f:
                json.dump(evaluation_log, f)
        except Exception as e:
            print(e)
        
        
        
    
    




















