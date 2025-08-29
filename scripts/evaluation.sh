#! /bin/bash

# evaluate best-of-N improvement
python -m evaluation.evaluate_bon_solution_stdio \
    --split test \
    --test_generation_model qwen3_4b_utrl_testgen_iter_2 \
    --solution_generation_model qwen3_4b \
    --best_of_n \
    --n_samples 32 \

python -m evaluation.evaluate_bon_solution_stdio \
    --split test \
    --test_generation_model qwen3_4b_utrl_testgen_iter_2 \
    --solution_generation_model qwen3_8b \
    --best_of_n \
    --n_samples 32 \

python -m evaluation.evaluate_bon_solution_stdio \
    --split test \
    --test_generation_model qwen3_4b_utrl_testgen_iter_2 \
    --solution_generation_model qwen3_14b \
    --best_of_n \
    --n_samples 32 \

python -m evaluation.evaluate_bon_solution_stdio \
    --split test \
    --test_generation_model qwen3_4b_utrl_testgen_iter_2 \
    --solution_generation_model gpt_4o \
    --best_of_n \
    --n_samples 32 \


# Show the Best-of-N improvement and Unit test Fidelity
python -m evaluation.summarize_evaluation.py \
    --test_generation_model qwen3_4b_utrl_testgen_iter_2 \

