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
def add_test_generation_prompt_for_test(example: Dict):
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
    example['data_source'] = 'testset'
    return example


def main():
    # 1. load dataset
    dataset = load_dataset(
        'BAAI/TACO',
        cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
        )
    
    # sampled code pool for training testgen model at iteration 2 
    with open("results/inference/taco_solutions/solution_by_qwen3_4b_utrl_codegen_iter_1_best_of_8_taco_train_split.json", 'r') as f:
        code_solutions = json.load(f)
        code_solutions_iter_1_train = []
        for code_solution in code_solutions:
            code_solution = random.sample([extract_python_code(c) for c in code_solution], 8)
            code_solutions_iter_1_train.append(code_solution)

    with open("results/inference/taco_solutions/solution_by_qwen3_4b_utrl_codegen_iter_1_best_of_8_taco_test_split.json", 'r') as f:
        code_solutions = json.load(f)
        code_solutions_iter_1_test = []
        for code_solution in code_solutions:
            code_solution = random.sample([extract_python_code(c) for c in code_solution], 8)
            code_solutions_iter_1_test.append(code_solution)

    # Add ground truth column
    train_dataset = dataset['train']
    train_dataset = train_dataset.add_column("ground_truth", code_solutions_iter_1_train)
    test_dataset = dataset['test']
    test_dataset = test_dataset.add_column("ground_truth", code_solutions_iter_1_test)
    
    # filter faulty samples
    train_dataset = train_dataset.filter(lambda x: "fn_name" not in json.loads(x['input_output']))
    
    # Postprocess function
    train_dataset = train_dataset.map(add_test_generation_prompt_for_train)
    test_dataset = test_dataset.map(add_test_generation_prompt_for_test)
    
    # Filter out samples with ground truth solution
    train_dataset = train_dataset.filter(lambda x: x['extra_info']['gt_solution'] is not None)
    # select necessary columns
    
    train_dataset = train_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    test_dataset = test_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    train_dataset, val_dataset = train_dataset.train_test_split(test_size=1000, seed=42)
    
    # save dataset
    train_dataset.to_parquet(os.path.join("data/", "train_testgen_iter_2_taco.parquet"))
    val_dataset.to_parquet(os.path.join("data/", "val_testgen_iter_2_taco.parquet"))
    test_dataset.to_parquet(os.path.join("data/", "eval_testgen_iter_2_taco.parquet"))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_for_code_sampling", type=str, default="qwen3_4b_utrl_codegen_iter_1")
    args = parser.parse_args()
    main(args)
