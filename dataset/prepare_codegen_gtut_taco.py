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
    
    example["extra_info"] = {
        'gt_solution': solution,
        }
    example["ground_truth"] = example['input_output']
    
    example['data_source'] = 'trainset'
    return example


# function to add chat templated SOLUTION GENERATION PROMPT
def add_solution_generation_prompt_for_test(example: Dict):
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
    example["extra_info"] = {
        'gt_solution': solution,
        }
    
    example["ground_truth"] = example['input_output']
    
    example['data_source'] = 'testset'
    return example



def main():
    # 1. load dataset
    dataset = load_dataset(
        'BAAI/TACO',
        cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
        )
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # filter faulty samples
    train_dataset = train_dataset.filter(lambda x: "fn_name" not in json.loads(x['input_output']))
    
    # Postprocess function
    train_dataset = train_dataset.map(add_solution_generation_prompt_for_train)
    test_dataset = test_dataset.map(add_solution_generation_prompt_for_test)
    
    # Filter out samples with ground truth solution
    train_dataset = train_dataset.filter(lambda x: x['extra_info']['gt_solution'] is not None)
    
    # select necessary columns
    train_dataset = train_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    test_dataset = test_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    train_dataset, val_dataset = train_dataset.train_test_split(test_size=1000, seed=42)
    # save dataset
    train_dataset.to_parquet(os.path.join("data", "train_codegen_gtut_taco.parquet"))
    test_dataset.to_parquet(os.path.join("data", "eval_codegen_gtut_taco.parquet"))
    val_dataset.to_parquet(os.path.join("data", "val_codegen_gtut_taco.parquet"))
if __name__ == "__main__":
    main()