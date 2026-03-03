import argparse
import json
import os
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from openai import OpenAI
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set cache directory to avoid permission issues
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/.cache/huggingface/datasets")

from utils.prompt import SOLUTION_GENERATION_PROMPT_STDIO as user_prompt
from utils.prompt import SOLUTION_GENERATION_SYSTEM_PROMPT_STDIO as system_prompt

HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/datasets")
OUTPUT_DIR = "outputs/inference/code_solutions"


class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        if self.max_requests_per_minute <= 0:
            return

        with self.lock:
            now = time.time()
            while self.requests and now - self.requests[0] >= 60:
                self.requests.popleft()

            if len(self.requests) >= self.max_requests_per_minute:
                sleep_time = 60 - (now - self.requests[0]) + 0.05
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.requests.append(time.time())


def get_output_path(args: argparse.Namespace) -> str:
    if args.best_of_n:
        return (
            f"{OUTPUT_DIR}/solution_by_{args.target_path}_best_of_{args.n_samples}_"
            f"{args.benchmark}_{args.split}_split.json"
        )
    return f"{OUTPUT_DIR}/solution_by_{args.target_path}_{args.benchmark}_{args.split}_split.json"


def load_benchmark_dataset(args: argparse.Namespace) -> Dataset:
    if args.benchmark == "taco":
        dataset_name = "dgjun32/UTRL_TACO_EVAL"
    else:
        dataset_name = "dgjun32/UTRL_LCB_EVAL"

    repo_path = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        cache_dir=HF_CACHE_DIR,
    )

    split_name = args.split if args.split else "train"
    split_pattern = f"{repo_path}/data/{split_name}-*.parquet"

    try:
        return load_dataset(
            "parquet",
            data_files=split_pattern,
            split="train",
        )
    except Exception:
        return load_dataset(
            "parquet",
            data_files=f"{repo_path}/data/train-*.parquet",
            split="train",
        )


def build_messages(
    dataset: Dataset,
    user_prompt: str,
    system_prompt: str,
    problem_key: str = "problem_statement",
) -> List[List[dict]]:
    print("Preparing messages...")
    all_messages: List[List[dict]] = []
    for i in range(len(dataset)):
        problem_query = dataset[i][problem_key]
        all_messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(problem_query=problem_query)},
            ]
        )
    return all_messages


def request_with_retry(
    client: OpenAI,
    rate_limiter: RateLimiter,
    model: str,
    message: List[dict],
    temperature: float,
    n: int,
    max_retries: int,
    retry_base_delay: float,
) -> Union[str, List[str]]:
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            response = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature,
                n=n,
            )
            if n == 1:
                return response.choices[0].message.content or ""

            outputs = [(choice.message.content or "") for choice in response.choices]
            if len(outputs) < n:
                outputs.extend([""] * (n - len(outputs)))
            return outputs
        except Exception as error:
            if attempt == max_retries - 1:
                print(f"Request failed after {max_retries} attempts: {error}")
                return ""
            delay = retry_base_delay * (1.5 ** attempt) + random.uniform(0.0, 0.1)
            time.sleep(delay)

    return ""


def generate_solutions_openai(
    args: argparse.Namespace,
    dataset: Dataset,
    client: OpenAI,
    rate_limiter: RateLimiter,
    user_prompt: str,
    system_prompt: str,
) -> Union[List[str], List[List[str]]]:
    all_messages = build_messages(dataset, user_prompt, system_prompt)
    all_outputs: Union[List[str], List[List[str]]] = []
    output_path = get_output_path(args)

    desc = "Generating multiple solutions with OpenAI API" if args.best_of_n else "Generating solutions with OpenAI API"
    n_samples = args.n_samples if args.best_of_n else 1

    def process_message(message: List[dict]) -> Union[str, List[str]]:
        if n_samples == 1:
            return request_with_retry(
                client=client,
                rate_limiter=rate_limiter,
                model=args.solution_generation_model,
                message=message,
                temperature=0.0,
                n=1,
                max_retries=args.max_retries,
                retry_base_delay=args.retry_base_delay,
            )

        candidates = request_with_retry(
            client=client,
            rate_limiter=rate_limiter,
            model=args.solution_generation_model,
            message=message,
            temperature=args.temperature,
            n=n_samples,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        if isinstance(candidates, list):
            return candidates
        return [candidates] + [""] * (n_samples - 1)

    for i in tqdm(range(0, len(all_messages), args.batch_size), desc=desc):
        batch_end = min(i + args.batch_size, len(all_messages))
        batch_messages = all_messages[i:batch_end]

        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(batch_messages))) as executor:
            batch_outputs = list(executor.map(process_message, batch_messages))

        all_outputs.extend(batch_outputs)

        with open(output_path, "w") as file:
            json.dump(all_outputs, file)

    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=["taco", "livecodebench"], required=True)
    parser.add_argument("--solution_generation_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--target_path", type=str, required=True, help="Model signature used in output file naming")
    parser.add_argument("--best_of_n", action="store_true", required=False)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_workers", type=int, default=16, help="Max parallel API workers per batch")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default=None, help="Optional custom endpoint (e.g., OpenRouter)")
    parser.add_argument("--max_requests_per_minute", type=int, default=0, help="0 disables client-side rate limiting")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_base_delay", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0, help="Used only when --best_of_n is enabled")
    args = parser.parse_args()

    if args.best_of_n and args.n_samples < 1:
        raise ValueError("--n_samples must be >= 1 when --best_of_n is enabled")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_file = get_output_path(args)

    dataset = load_benchmark_dataset(args)

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    rate_limiter = RateLimiter(max_requests_per_minute=args.max_requests_per_minute)

    if not args.best_of_n and os.path.exists(cache_file):
        print("Loading solutions from cache...")
        with open(cache_file, "r") as file:
            solutions = json.load(file)
    else:
        print("Generating multiple solutions..." if args.best_of_n else "Generating solutions...")
        solutions = generate_solutions_openai(
            args=args,
            dataset=dataset,
            client=client,
            rate_limiter=rate_limiter,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        with open(cache_file, "w") as file:
            json.dump(solutions, file)

    print(f"Generation complete! Generated {len(solutions)} solutions.")
    if args.best_of_n:
        total_samples = sum(len(solution_list) for solution_list in solutions)
        print(f"Total samples generated: {total_samples}")
