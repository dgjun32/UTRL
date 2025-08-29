import os
import random
from math import inf
from typing import Dict
import json
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.model_selection import train_test_split

from utils.prompt import SOLUTION_GENERATION_PROMPT_STDIO, SOLUTION_GENERATION_SYSTEM_PROMPT_STDIO
from utils.parsing_utils import extract_test_cases_stdio
from utils.testing_utils import run_testcase_stdio


# function to add chat templated SOLUTION GENERATION PROMPT
def add_solution_generation_prompt_for_train(example: Dict):
    user_prompt= SOLUTION_GENERATION_PROMPT_STDIO.format(problem_query = example["question"])
    system_prompt = SOLUTION_GENERATION_SYSTEM_PROMPT_STDIO
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
        
    # get GT test cases
    """
    try:
        gt_test = []
        test_cases = eval(example['input_output'])
        for inp, out in zip(test_cases['inputs'], test_cases['outputs']):
            gt_test.append({'input': inp, 'output': out})
    except:
        gt_test = []  # Changed from None to empty list
    """
    example["extra_info"] = {
        'gt_solution': solution,
        }
    example["ground_truth"] = example['input_output']
    
    example['data_source'] = 'trainset'
    return example


# function to add chat templated SOLUTION GENERATION PROMPT
def add_solution_generation_prompt_for_validation(example: Dict):
    user_prompt= SOLUTION_GENERATION_PROMPT_STDIO.format(problem_query = example["question"])
    system_prompt = SOLUTION_GENERATION_SYSTEM_PROMPT_STDIO
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
        
    # get GT test cases
    """
    try:
        gt_test = []
        test_cases = eval(example['input_output'])
        if len(test_cases['inputs']) == 0:
            test_cases = example['public_tests']
        for inp, out in zip(test_cases['inputs'], test_cases['outputs']):
            gt_test.append({'input': inp, 'output': out})
    except:
        gt_test = []  # Changed from None to empty list
    """
    example["extra_info"] = {
        'gt_solution': solution,
        }
    
    example["ground_truth"] = example['input_output']
    
    example['data_source'] = 'validationset'
    return example



def main():
    # 1. load dataset
    dataset = load_dataset(
        'BAAI/TACO',
        cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
        )
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    # filter faulty samples
    train_dataset = train_dataset.filter(lambda x: "fn_name" not in json.loads(x['input_output']))
    
    # Postprocess function
    train_dataset = train_dataset.map(add_solution_generation_prompt_for_train)
    val_dataset = val_dataset.map(add_solution_generation_prompt_for_validation)
    
    # Filter out samples with ground truth solution
    train_dataset = train_dataset.filter(lambda x: x['extra_info']['gt_solution'] is not None)
    
    # select necessary columns
    train_dataset = train_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    eval_dataset = val_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    # save dataset
    breakpoint()
    train_dataset.to_parquet(os.path.join("data", "train_codegen_gtut_taco.parquet"))
    eval_dataset.to_parquet(os.path.join("data", "eval_codegen_gtut_taco.parquet"))
    
if __name__ == "__main__":
    main()