import os
from math import inf
import random
from typing import Dict
import ast
import json
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from utils.prompt import TEST_GENERATION_PROMPT_STDIO, TEST_GENERATION_SYSTEM_PROMPT_STDIO
from utils.parsing_utils import extract_python_code
from utils.testing_utils import run_testcase


# function to add chat templated SOLUTION GENERATION PROMPT
def add_test_generation_prompt_for_train(example: Dict):
    user_prompt= TEST_GENERATION_PROMPT_STDIO.format(problem_query = example["question"])
    system_prompt = TEST_GENERATION_SYSTEM_PROMPT_STDIO
    example["prompt"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # get Python GT solution
    try:
        solutions = json.loads(example['solutions'])
        if len(solutions) == 0:
            solution = None
        else:
            solution = random.choice(solutions)
    except:
        solution = None
    example["extra_info"] = {
        'gt_solution': solution
        }
    example['data_source'] = 'trainset'
    return example


# function to add chat templated SOLUTION GENERATION PROMPT
def add_test_generation_prompt_for_validation(example: Dict):
    user_prompt= TEST_GENERATION_PROMPT_STDIO.format(problem_query = example["question"])
    system_prompt = TEST_GENERATION_SYSTEM_PROMPT_STDIO
    # get Python GT solution
    example["prompt"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # get Python GT solution
    try:
        solutions = json.loads(example['solutions'])
        if len(solutions) == 0:
            solution = None
        else:
            solution = random.choice(solutions)
    except:
        solution = None
    example["extra_info"] = {
        'gt_solution': solution
        }
    example['data_source'] = 'validationset'
    return example


def main():
    # 1. load dataset
    dataset = load_dataset(
        'BAAI/TACO',
        cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
        )
    
    # sampled code pool for training testgen model at iteration 2 
    with open("results/inference/taco_solutions/solution_by_qwen3_4b_codegen_iter_1_step_370_best_of_8_taco_train_split.json", 'r') as f:
        code_solutions = json.load(f)
        code_solutions_iter_1_train = []
        for code_solution in code_solutions:
            code_solution = random.sample([extract_python_code(c) for c in code_solution], 8)
            code_solutions_iter_1_train.append(code_solution)

    with open("results/inference/taco_solutions/solution_by_qwen3_4b_codegen_iter_1_step_370_best_of_8_taco_test_split.json", 'r') as f:
        code_solutions = json.load(f)
        code_solutions_iter_1_test = []
        for code_solution in code_solutions:
            code_solution = random.sample([extract_python_code(c) for c in code_solution], 8)
            code_solutions_iter_1_test.append(code_solution)
    
    # sampled code pool for training testgen model at iteration 1 (previous iteration)
    iter_1_train_dataset = pd.read_parquet("data/train_testgen_iter_1_taco.parquet")
    code_solutions_iter_0_train = iter_1_train_dataset['ground_truth'].tolist()
    iter_1_test_dataset = pd.read_parquet('data/eval_testgen_iter_1_taco.parquet')
    code_solutions_iter_0_test = iter_1_test_dataset['ground_truth'].tolist()

    # Add ground truth column
    train_dataset = dataset['train']
    train_dataset = train_dataset.add_column("ground_truth", code_solutions_iter_1_train)
    val_dataset = dataset['test']
    val_dataset = val_dataset.add_column("ground_truth", code_solutions_iter_1_test)
    
    # filter faulty samples
    train_dataset = train_dataset.filter(lambda x: "fn_name" not in json.loads(x['input_output']))
    #val_dataset = val_dataset.filter(lambda x: "fn_name" not in json.loads(x['input_output']))
    
    # Postprocess function
    train_dataset = train_dataset.map(add_test_generation_prompt_for_train)
    val_dataset = val_dataset.map(add_test_generation_prompt_for_validation)
    
    # Filter out samples with ground truth solution
    train_dataset = train_dataset.filter(lambda x: x['extra_info']['gt_solution'] is not None)
    #val_dataset = val_dataset.filter(lambda x: x['extra_info']['gt_solution'] is not None)
    # select necessary columns
    
    train_dataset = train_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    eval_dataset = val_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    
    # add previous iteration's code solutions
    code_solutions_iter_1_train = train_dataset['ground_truth']
    code_solutions_iter_1_test = eval_dataset['ground_truth']
    code_solutions_train, code_solutions_test = [], []
    
    for iter_0, iter_1 in zip(code_solutions_iter_0_train, code_solutions_iter_1_train):
        code_solutions_train.append({'iter_0': iter_0, 'iter_1': iter_1})
    
    for iter_0, iter_1 in zip(code_solutions_iter_0_test, code_solutions_iter_1_test):
        code_solutions_test.append({'iter_0': iter_0, 'iter_1': iter_1})
    
    # 기존 ground_truth 컬럼 제거 후 새로운 컬럼 추가
    train_dataset = train_dataset.remove_columns(["ground_truth"]).add_column("ground_truth", code_solutions_train)
    train_dataset = train_dataset.filter(lambda x: "class Solution" not in x['extra_info']['gt_solution'] and "print(" in x['extra_info']['gt_solution'])
    eval_dataset = eval_dataset.remove_columns(["ground_truth"]).add_column("ground_truth", code_solutions_test)
    # save dataset
    breakpoint()
    #train_dataset.to_parquet(os.path.join("mnt/dongjunlee/data/", "train_testgen_iter_2_taco.parquet"))
    train_dataset.to_parquet(os.path.join("data/", "train_testgen_iter_2_taco_370.parquet"))
    #eval_dataset.to_parquet(os.path.join("mnt/dongjunlee/data/", "eval_testgen_iter_2_taco.parquet"))
    eval_dataset.to_parquet(os.path.join("data/", "eval_testgen_iter_2_taco_370.parquet"))
    
if __name__ == "__main__":
    main()
