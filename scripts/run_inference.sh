python -m inference.generate_unit_tests_with_api --benchmark taco \
    --solution_generation_model gpt-4.1 \
    --target_path gpt_4_1 \
    --api_key $OPENAI_API_KEY


python -m inference.generate_unit_tests --benchmark taco \
    --solution_generation_model Qwen/Qwen3-14B \
    --target_path qwen3_14b \