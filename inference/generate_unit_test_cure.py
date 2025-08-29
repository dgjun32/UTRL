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
from transformers import AutoTokenizer

from utils.llm_utils import VLLMGenerator



def generate_multiple_tests(
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
    # Use larger batch size for vLLM (it handles batching efficiently)
    batch_size = 512  # Much larger than before due to vLLM efficiency
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
    for i in tqdm(range(0, len(all_messages), batch_size), desc="Generating multiple unit tests with vLLM"):
        batch_end = min(i + batch_size, len(all_messages))
        batch_messages = all_messages[i:batch_end]
        
        # Generate solutions
        batch_solutions = vllm_generator.generate(batch_messages, n_samples)
        all_solutions.extend(batch_solutions)
        
        # Save periodically
        with open(f"results/inference/taco_unit_tests/unit_test_by_{args.target_path}_best_of_{args.n_samples}_taco_{args.split}_split.json", "w") as f:
            json.dump(all_solutions, f)
        
    return all_solutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_generation_model", type=str, default="GenVerse/ReasonFlux-Coder-7B")
    parser.add_argument("--target_path", type=str, help="The signature of the model checkpoint")
    parser.add_argument("--best_of_n", action="store_true", required=False)
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("results/inference/taco_unit_tests/", exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    from utils.prompt import TEST_GENERATION_PROMPT_CURE as user_prompt
    from utils.prompt import TEST_GENERATION_SYSTEM_PROMPT_CURE as system_prompt
    dataset = load_dataset(
        "BAAI/TACO",
        cache_dir=os.path.expanduser('~/.cache/huggingface/datasets'),
        trust_remote_code=True
    )[args.split]
    
    # Initialize vLLM generator
    print(f"Initializing vLLM with model: {args.test_generation_model}")
    vllm_generator = VLLMGenerator(
        model_name=args.test_generation_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=32768
    )
    
    cache_file = f"results/inference/unit_test_by_{args.target_path}_best_of_{args.n_samples}_{args.benchmark}_{args.split}_split.json"
        
    print("Generating test scripts...")
    tests = generate_multiple_tests(
        args=args,
        dataset=dataset,
        vllm_generator=vllm_generator,
        n_samples=args.n_samples,
        user_prompt = user_prompt,
        system_prompt = system_prompt
    )