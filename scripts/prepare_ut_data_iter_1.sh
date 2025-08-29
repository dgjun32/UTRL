#! /bin/bash

# 1. Sample code for each programming task via LLM
python -m inference.generate_solution \
    --solution_generation_model Qwen/Qwen3-4B \
    --target_path qwen3_4b \
    --best_of_n \
    --n_samples 8 \
    --split train \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.9 \
    --batch_size 128

python -m inference.generate_solution \
    --solution_generation_model Qwen/Qwen3-4B \
    --target_path qwen3_4b \
    --best_of_n \
    --n_samples 8 \
    --split test \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.9 \
    --batch_size 128


# 2. Create dataset
python -m dataset.prepare_testgen_iter_1_taco \
    --model_for_code_sampling qwen3_4b \