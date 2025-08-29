python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_50/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_50/actor_hf \

echo "Step 50 converted"

python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_100/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_100/actor_hf \

echo "Step 100 converted"

python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_150/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_150/actor_hf \

echo "Step 150 converted"

python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_200/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_200/actor_hf \

echo "Step 200 converted"

python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_50/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_50/actor_hf \

echo "Step 200 converted"

python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_250/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_250/actor_hf \

echo "Step 250 converted"

python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_300/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_300/actor_hf \

echo "Step 300 converted"

python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_350/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_350/actor_hf \

echo "Step 350 converted"

python scripts/model_merger.py merge --backend fsdp \
 --local_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_400/actor \
 --target_dir ../mnt/dongjunlee/qwen3_4b_codegen_iter_1/global_step_400/actor_hf \

echo "Step 400 converted"