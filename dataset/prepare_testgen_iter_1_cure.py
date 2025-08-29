



import os
from math import inf
import random
from typing import Dict
import ast
import json
from tqdm import tqdm
import datasets
from datasets import load_dataset, concatenate_datasets
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import functools

from utils.prompt import TEST_GENERATION_PROMPT_STDIO, TEST_GENERATION_SYSTEM_PROMPT_STDIO
from utils.parsing_utils import extract_test_cases_v2, extract_python_code
from utils.testing_utils import run_testcase_stdio


# function to add chat templated SOLUTION GENERATION PROMPT
def add_test_generation_prompt_for_train(example: Dict):
    user_prompt= TEST_GENERATION_PROMPT_STDIO.format(problem_query = example["question"])
    system_prompt = TEST_GENERATION_SYSTEM_PROMPT_STDIO
    example["prompt"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return example


from utils.testing_utils import run_testcase_stdio

def main():
    # 1. load dataset
    with open('/home/dongjun/utrl/results/evaluation/CodeContests_train_updated.json', 'r') as f:
        dataset = json.load(f)
    updated_dataset = []
    
    with open("results/inference/codecontests_cure_solutions/solution_by_qwen2.5_7b_best_of_8.json", 'r') as f:
        code_solutions_train = json.load(f)
        code_solutions_train_extracted = []
        for code_solutions in code_solutions_train:
            code_solutions = random.sample([extract_python_code(c) for c in code_solutions], 8)
            code_solutions_train_extracted.append(code_solutions)
    
    def run_single_test(sol, test_pair):
        inp, out = test_pair
        return run_testcase_stdio(sol, {'input': inp, 'output': out})

    
    cnt_failed_instance = 0
    for i, (instance, sampled_solutions) in enumerate(tqdm(zip(dataset, code_solutions_train_extracted))):
        instance['ground_truth'] = sampled_solutions
        
        # choose working ground-truth solution
        for j, sol in enumerate(instance['solutions']):
            n_passed = 0
            n_tests = len(instance['test_input'])
            test_pairs = list(zip(instance['test_input'], instance['test_output']))

            with ThreadPoolExecutor(max_workers=32) as executor:  # I/O 작업이므로 더 많은 쓰레드 가능
                results = executor.map(
                    functools.partial(run_single_test, sol), 
                    test_pairs
                )
                n_passed = sum(1 for result in results if result['passed'])
                
            if n_passed == n_tests:
                instance['extra_info'] = {'gt_solution': sol}
                instance['data_source'] = 'trainset'
                updated_dataset.append(instance)
                break
            
            if j == len(instance['solutions']) - 1:
                cnt_failed_instance += 1 
        
        print("Ratio of failed instances until now: ", cnt_failed_instance / (i + 1))
        print("Currently processed: ", i + 1)
        print("Successfully processed: ", len(updated_dataset))
    breakpoint()
    train_dataset = datasets.Dataset.from_list(updated_dataset)
    
    # filter faulty samples
    #train_dataset = train_dataset.filter(lambda x: "fn_name" not in json.loads(x['input_output']))
    #val_dataset = val_dataset.filter(lambda x: "fn_name" not in json.loads(x['input_output']))
    #breakpoint()
    
    # Postprocess function
    train_dataset = train_dataset.map(add_test_generation_prompt_for_train)
    #val_dataset = val_dataset.filter(lambda x: x['extra_info']['gt_solution'] is not None)
    # select necessary columns
    #breakpoint()
    
    train_dataset = train_dataset.select_columns(["prompt", "extra_info", "data_source", "ground_truth"])
    # save dataset
    breakpoint()
    train_dataset.to_parquet(os.path.join("data/", "train_testgen_iter_1_cure.parquet"))
    
if __name__ == "__main__":
    main()
