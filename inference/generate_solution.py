import re
import base64
import random
import os
import ast 
import astor
import json
import math
import tempfile
import subprocess
import time
import concurrent
import argparse
import asyncio
import sys
from typing import List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set cache directory to avoid permission issues
os.environ['HF_DATASETS_CACHE'] = os.path.expanduser('~/.cache/huggingface/datasets')

import tqdm
from tqdm import tqdm
import torch
import datasets
import transformers
import numpy as np
import openai
from openai import AzureOpenAI
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.distributed import init_distributed_environment

from utils.llm_utils import VLLMGenerator


def generate_multiple_solutions(
    args: argparse.Namespace,
    dataset: Dataset,
    vllm_generator: VLLMGenerator,
    n_samples: int,
    user_prompt: str,
    system_prompt: str,
) -> List[List[str]]:
    """
    Optimized multiple solution generation using vLLM
    """
    batch_size = args.batch_size
    all_solutions = []
    problem_key = "question"
    
    # Pre-build all messages to avoid repeated computation
    print("Preparing messages...")
    all_messages = []
    for i in range(len(dataset)):
        problem_query = dataset[i][problem_key]
        message = [
                {
                "role": "system",
                "content": system_prompt
                },
                {
                "role": "user",
                "content": user_prompt.format(problem_query=problem_query)
                }
            ]
        all_messages.append(message)
    
    # Process in batches
    for i in tqdm(range(0, len(all_messages), batch_size), desc="Generating multiple solutions with vLLM"):
        batch_end = min(i + batch_size, len(all_messages))
        batch_messages = all_messages[i:batch_end]
        
        # Generate solutions
        batch_solutions = vllm_generator.generate(batch_messages, n_samples)
        all_solutions.extend(batch_solutions)
        # Save periodically
        with open(f"results/inference/taco_solutions/solution_by_{args.target_path}_best_of_{args.n_samples}_taco_{args.split}_split.json", "w") as f:
            json.dump(all_solutions, f)
        
        
    return all_solutions


def generate_solution(
    args: argparse.Namespace,
    dataset: Dataset,
    vllm_generator: VLLMGenerator,
    user_prompt: str,
    system_prompt: str,
) -> List[str]:
    """
    Optimized single solution generation using vLLM
    """
    batch_size = args.batch_size
    all_responses = []
    problem_key = "question"
        
    # Pre-build all messages
    print("Preparing messages...")
    all_messages = []
    for i in range(len(dataset)):
        problem_query = dataset[i][problem_key]
        message = [
                {
                "role": "system",
                "content": system_prompt
                },
                {
                "role": "user",
                "content": user_prompt.format(problem_query=problem_query)
                }
            ]
        all_messages.append(message)
    
    # Process in large batches
    for i in tqdm(range(0, len(all_messages), batch_size), desc="Generating solutions with vLLM"):
        batch_end = min(i + batch_size, len(all_messages))
        batch_messages = all_messages[i:batch_end]
        
        # Generate solutions (single sample)
        batch_solutions = vllm_generator.generate(batch_messages, n_samples=1)
        # Extract single solutions from nested lists
        batch_responses = [sol[0] for sol in batch_solutions]
        all_responses.extend(batch_responses)
        # Save periodically
        with open(f"results/inference/taco_solutions/solution_by_{args.target_path}_best_of_{args.n_samples}_taco_{args.split}_split.json", "w") as f:
            json.dump(all_responses, f)
    
    return all_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_generation_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--target_path", type=str, help="The signature of the solution generation model")
    parser.add_argument("--best_of_n", action="store_true", required=False)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(f"results/inference/taco_solutions/", exist_ok=True)
    
    # set benchmark configs
    dataset_name = "BAAI/TACO"
    from utils.prompt import SOLUTION_GENERATION_PROMPT_STDIO as user_prompt
    from utils.prompt import SOLUTION_GENERATION_SYSTEM_PROMPT_STDIO as system_prompt
    dataset = load_dataset(
        dataset_name,
        cache_dir=os.path.expanduser('~/.cache/huggingface/datasets'),
        trust_remote_code=True
    )[args.split]
    
    # Initialize vLLM generator
    print(f"Initializing vLLM with model: {args.solution_generation_model}")
    vllm_generator = VLLMGenerator(
        model_name=args.solution_generation_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Generate solutions
    if not args.best_of_n:
        # Single solution per problem
        cache_file = f"results/inference/taco_solutions/solution_by_{args.target_path}_taco_{args.split}_split.json"
        if os.path.exists(cache_file):
            print("Loading solutions from cache...")
            solutions = json.load(open(cache_file))
        else:
            print("Generating solutions...")
            solutions = generate_solution(
                args=args,
                dataset=dataset,
                vllm_generator=vllm_generator,
                user_prompt = user_prompt,
                system_prompt = system_prompt
            )
            
            # Final save
            with open(cache_file, "w") as f:
                json.dump(solutions, f)
                
    elif args.best_of_n:
        # Multiple solutions per problem
        cache_file = f"results/inference/taco_solutions/solution_by_{args.target_path}_best_of_{args.n_samples}_taco_{args.split}_split.json"
        print("Generating multiple solutions...")
        solutions = generate_multiple_solutions(
            args=args,
            dataset=dataset,
            vllm_generator=vllm_generator,
            n_samples=args.n_samples,
            user_prompt = user_prompt,
            system_prompt = system_prompt
        )
        # Final save
        with open(cache_file, "w") as f:
            json.dump(solutions, f)

    print(f"Generation complete! Generated {len(solutions)} solutions.")
    if args.best_of_n:
        total_samples = sum(len(sol_list) for sol_list in solutions)
        print(f"Total samples generated: {total_samples}") 