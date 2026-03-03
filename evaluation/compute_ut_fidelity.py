import argparse
import json
import math
import os
from typing import List, Dict

from scipy.stats import spearmanr


BASE_DIR = "./outputs/evaluation/best_of_n_improvement"
SOLUTION_MODELS = ["gpt_4o", "qwen3_4b", "qwen3_8b", "qwen3_14b"]


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        return json.load(file)


def build_file_path(test_model: str, solution_model: str, benchmark: str) -> str:
    return (
        f"{BASE_DIR}/"
        f"{benchmark}_test_by_{test_model}_"
        f"solution_by_{solution_model}_"
        f"best_of_32_{benchmark}_test_split.json"
    )


def main(args: argparse.Namespace) -> None:
    gt_by_model = {}
    pred_by_model = {}

    for solution_model in SOLUTION_MODELS:
        gt_path = build_file_path("ground_truth", solution_model, args.benchmark)
        pred_path = build_file_path(args.test_generation_model, solution_model, args.benchmark)

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground-truth file: {gt_path}")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")

        gt_by_model[solution_model] = load_json(gt_path)
        pred_by_model[solution_model] = load_json(pred_path)
        
    # sum([x['passed'] for x in pred_by_model['qwen3_14b']])/945
    # sum([x['score'] for x in pred_by_model['qwen3_14b']])/945
    lengths = [len(gt_by_model[m]) for m in SOLUTION_MODELS] + [len(pred_by_model[m]) for m in SOLUTION_MODELS]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent file lengths: {lengths}")

    n_tasks = lengths[0]
    correlations: List[float] = []

    for i in range(n_tasks):
        gt_vector: List[float] = []
        pred_vector: List[float] = []

        for solution_model in SOLUTION_MODELS:
            gt_scores = gt_by_model[solution_model][i]["measured_scores"]
            pred_scores = pred_by_model[solution_model][i]["measured_scores"]

            gt_vector.extend(gt_scores)
            pred_vector.extend(pred_scores)

        if len(gt_vector) != len(pred_vector):
            raise ValueError(f"Vector length mismatch at index {i}: {len(gt_vector)} vs {len(pred_vector)}")
        
        corr, _ = spearmanr(gt_vector, pred_vector)
        if not math.isnan(corr):
            correlations.append(corr)
        else:
            correlations.append(0.0)

    if not correlations:
        raise RuntimeError("All task correlations are NaN. Check measured_scores vectors.")

    mean_corr = sum(correlations) / len(correlations)

    print("=" * 80)
    print(f"test_generation_model: {args.test_generation_model}")
    print(f"num_tasks_total: {n_tasks}")
    print(f"num_tasks_valid: {len(correlations)}")
    print(f"mean_spearman_corr: {mean_corr:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=["taco", "livecodebench"], default="taco")
    parser.add_argument("--test_generation_model", type=str, required=True)
    parsed_args = parser.parse_args()
    main(parsed_args)
