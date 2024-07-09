#!/bin/bash
#SBATCH --partition=a100          # Specify the partition or queue
#SBATCH --gres=gpu:a100:8         # Request specific GPU resources
#SBATCH --time=23:50:00           # Set a limit on the total run time
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --job-name=peft-adapter-32-1
#SBATCH --mail-type=ALL
#SBATCH --constraint=a100_80

source .bashrc
python3 llava/llava_PEFT_init.py


deepspeed llava/train_parallel_deepspeed_mixtral_adapter.py --num_stages=8 \
   --shared_adapter_num=1 --shared_adapter_type=Parallel_Adapter --hidden_dim=32 \
   --save_model_shard=100  --skip_shard=1300  \
   --checkpoint_dir=/home/atuin/b207dd/b207dd11/LLaVA-PEFT_adapter_32_64_checkpoint 2>&1 | tee llava/train_parallel_deepspeed_mixtral_adapter_32_64.log

echo -e "\a"
