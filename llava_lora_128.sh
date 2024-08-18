#!/bin/bash
#SBATCH --partition=a100          # Specify the partition or queue
#SBATCH --gres=gpu:a100:8         # Request specific GPU resources
#SBATCH --time=5:59:00           # Set a limit on the total run time
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --job-name=peft-lora-32-4
#SBATCH --mail-type=ALL
#SBATCH --constraint=a100_80

source .bashrc
python3 llava/llava_PEFT_init.py



deepspeed llava/train_parallel_deepspeed_mixtral_lora.py --num_stages=8 \
   --lora_r=128 --lora_alpha=256 \
   --save_model_shard=100  --skip_shard=860  \
   --checkpoint_dir=/home/atuin/b207dd/b207dd11/LLaVA-PEFT_lora_128_256_checkpoint 2>&1 | tee llava/train_parallel_deepspeed_mixtral_lora_128_256.log

echo -e "\a"

