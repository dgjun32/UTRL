import os
import json
import argparse
import sys
from typing import List, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set cache directory to avoid permission issues
os.environ['HF_DATASETS_CACHE'] = os.path.expanduser('~/.cache/huggingface/datasets')

from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
from huggingface_hub import snapshot_download

from utils.llm_utils import VLLMGenerator
from utils.prompt import SOLUTION_GENERATION_PROMPT_STDIO as user_prompt
from utils.prompt import SOLUTION_GENERATION_SYSTEM_PROMPT_STDIO as system_prompt


HF_CACHE_DIR = os.path.expanduser('~/.cache/huggingface/datasets')
OUTPUT_DIR = "outputs/inference/code_solutions"


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
    elif args.benchmark == "livecodebench":
        dataset_name = "dgjun32/UTRL_LCB_EVAL"
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

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


def build_messages(dataset: Dataset, user_prompt: str, system_prompt: str, problem_key: str = "problem_statement") -> List[List[dict]]:
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


def generate_solutions(
    args: argparse.Namespace,
    dataset: Dataset,
    vllm_generator: VLLMGenerator,
    n_samples: int,
    user_prompt: str,
    system_prompt: str,
) -> Union[List[str], List[List[str]]]:
    batch_size = args.batch_size
    all_outputs: Union[List[str], List[List[str]]] = []
    output_path = get_output_path(args)
    all_messages = build_messages(dataset, user_prompt, system_prompt)

    desc = "Generating multiple solutions with vLLM" if n_samples > 1 else "Generating solutions with vLLM"
    for i in tqdm(range(0, len(all_messages), batch_size), desc=desc):
        batch_end = min(i + batch_size, len(all_messages))
        batch_messages = all_messages[i:batch_end]

        batch_solutions = vllm_generator.generate(batch_messages, n_samples=n_samples)
        if n_samples == 1:
            all_outputs.extend([sol[0] for sol in batch_solutions])
        else:
            all_outputs.extend(batch_solutions)

        with open(output_path, "w") as f:
            json.dump(all_outputs, f)

    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=["taco", "livecodebench"], default="taco")
    parser.add_argument("--solution_generation_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--target_path", type=str, help="The signature of the solution generation model")
    parser.add_argument("--best_of_n", action="store_true", required=False)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dataset = load_benchmark_dataset(args)
    
    # Initialize vLLM generator
    print(f"Initializing vLLM with model: {args.solution_generation_model}")
    vllm_generator = VLLMGenerator(
        model_name=args.solution_generation_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Generate solutions
    cache_file = get_output_path(args)
    n_samples = args.n_samples if args.best_of_n else 1

    if not args.best_of_n and os.path.exists(cache_file):
            print("Loading solutions from cache...")
            solutions = json.load(open(cache_file))
    else:
        print("Generating multiple solutions..." if args.best_of_n else "Generating solutions...")
        solutions = generate_solutions(
            args=args,
            dataset=dataset,
            vllm_generator=vllm_generator,
            n_samples=n_samples,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        with open(cache_file, "w") as f:
            json.dump(solutions, f)

    print(f"Generation complete! Generated {len(solutions)} solutions.")
    if args.best_of_n:
        total_samples = sum(len(sol_list) for sol_list in solutions)
        print(f"Total samples generated: {total_samples}") 