#! /bin/bash
CKPT_STEP=$1

echo "Convert FSDP checkpoint of Code generator to HF checkpoint (using step ${CKPT_STEP})"
# convert FSDP checkpoint 
python scripts/model_merger.py merge --backend fsdp --local_dir ckpt/qwen3_4b_utrl_codegen_iter_1/global_step_${CKPT_STEP}/actor --target_dir ckpt/qwen3_4b_utrl_codegen_iter_1/global_step_${CKPT_STEP}/actor_hf

echo "Generate unit test using the checkpoint"
# 1. Generate unit test for each programming task via LLM
python -m inference.generate_solution \
    --test_generation_model ckpt/qwen3_4b_utrl_codegen_iter_1/global_step_${CKPT_STEP}/actor_hf \
    --target_path qwen3_4b_utrl_codegen_iter_1 \
    --best_of_n \
    --n_samples 8 \
    --split train \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.9 \
    --batch_size 128

python -m inference.generate_unit_test \
    --test_generation_model ckpt/qwen3_4b_utrl_iter_1/global_step_${CKPT_STEP}/actor_hf \
    --target_path qwen3_4b_utrl_iter_1 \
    --best_of_n \
    --n_samples 8 \
    --split test \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.9 \
    --batch_size 128


# 2. Create dataset for training code generator
python -m dataset.prepare_codegen_iter_1_taco \
    --model_for_ut_sampling qwen3_4b_utrl_iter_1 \