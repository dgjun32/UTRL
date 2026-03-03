python -m evaluation.evaluate_bon_solution --benchmark taco \
  --test_generation_model gpt_4_1 \
  --solution_generation_model qwen3_4b \
  --best_of_n \
  --n_samples 32

python -m evaluation.evaluate_bon_solution --benchmark taco \
  --test_generation_model gpt_4_1 \
  --solution_generation_model qwen3_8b \
  --best_of_n \
  --n_samples 32

python -m evaluation.evaluate_bon_solution --benchmark taco \
  --test_generation_model gpt_4_1 \
  --solution_generation_model qwen3_14b \
  --best_of_n \
  --n_samples 32

python -m evaluation.evaluate_bon_solution --benchmark taco \
  --test_generation_model gpt_4_1 \
  --solution_generation_model gpt_4o \
  --best_of_n \
  --n_samples 32

python -m evaluation.evaluate_bon_solution --benchmark taco \
  --test_generation_model ground_truth \
  --solution_generation_model gpt_4o \
  --best_of_n \
  --n_samples 32

python -m evaluation.compute_ut_fidelity --benchmark taco \
  --test_generation_model gpt_4_1