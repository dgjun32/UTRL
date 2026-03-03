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
import openai
from openai import AzureOpenAI
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


from utils.llm_utils import VLLMGenerator


def generate_test(
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
    problem_key = "problem_statement"
    
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
    for i in tqdm(range(0, len(all_messages), batch_size), desc="Generating unit tests with vLLM"):
        batch_end = min(i + batch_size, len(all_messages))
        batch_messages = all_messages[i:batch_end]
        
        # Generate solutions (single sample)
        batch_solutions = vllm_generator.generate(batch_messages, n_samples=1)
        # Extract single solutions from nested lists
        batch_responses = [sol[0] for sol in batch_solutions]
        all_responses.extend(batch_responses)

        # Save periodically
        with open(f'outputs/inference/unit_tests/unit_test_by_{args.target_path}_{args.benchmark}_{args.split}_split.json', "w") as f:
            json.dump(all_responses, f)
    
    return all_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=["taco", "livecodebench"], default="taco")
    parser.add_argument("--test_generation_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--target_path", type=str, help="The signature of the model checkpoint")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(f"outputs/inference/unit_tests/", exist_ok=True)
    
    # set benchmark configs
    if args.benchmark == "taco":
        dataset_name = "dgjun32/UTRL_TACO_EVAL"
    elif args.benchmark == "livecodebench":
        dataset_name = "dgjun32/UTRL_LCB_EVAL"
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")
    
    from utils.prompt import TEST_GENERATION_PROMPT_STDIO as user_prompt
    from utils.prompt import TEST_GENERATION_SYSTEM_PROMPT_STDIO as system_prompt
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
    
    # Initialize vLLM generator
    print(f"Initializing vLLM with model: {args.test_generation_model}")
    vllm_generator = VLLMGenerator(
        model_name=args.test_generation_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Single solution per problem
    cache_file = f'outputs/inference/unit_tests/unit_test_by_{args.target_path}_{args.benchmark}_{args.split}_split.json'
    print("Generating test scripts...")
    tests = generate_test(
        args=args,
        dataset=dataset,
        vllm_generator=vllm_generator,
        user_prompt = user_prompt,
        system_prompt = system_prompt
    )
        
    # Final save
    with open(cache_file, "w") as f:
        json.dump(tests, f)