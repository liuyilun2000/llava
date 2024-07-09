#module load cuda/12.1.1
#ds_report
'''
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
model = LlavaForConditionalGeneration.from_pretrained(
    '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/',
    device_map="cpu",
    torch_dtype="auto"
)
'''

#CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed llava/pretrain/train_parallel_deepspeed_mixtral.py --num_stages=4
#deepspeed llava/pretrain/train_parallel_deepspeed_mixtral.py --num_stages=8 2>&1 | tee llava/pretrain/train_parallel_deepspeed.log
'''
deepspeed llava/pretrain/train_parallel_deepspeed_mixtral.py \
    --checkpoint_dir=/home/atuin/b207dd/b207dd11/LLaVA-Pretrain_shard_0_checkpoint --num_stages=8 2>&1 | tee llava/pretrain/train_parallel_deepspeed_shard_0.log
'''


TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

DATASET_SIZE = 558128
NUM_DATASET_SHARDS = 437
MAP_BATCH_SIZE = 1280

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"


import os
import copy
import json
import math
import time
import pickle
import argparse
import requests
import numpy as np
from tqdm import tqdm

from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset, load_from_disk
#from huggingface_hub import upload_file, upload_folder

import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel

from transformers import Trainer, TrainingArguments
from transformers import AdamW
from transformers import get_scheduler

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe import ProcessTopology
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import comm as dist

from deepspeed_pipeline_model import *

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

def set_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num_stages", type=int, default=8, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--multi_modal_projector_pretraining", type=bool, default=True, help="")
    #parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")

    parser.add_argument("--dataloader_batch_size", type=int, default=2, help="")
    parser.add_argument("--train_batch_size", type=int, default=128, help="")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128, help="")

    parser.add_argument("--steps_per_print", default=1, type=int, help="")
    parser.add_argument("--save_model_shard", default=20, type=int, help="")
    parser.add_argument("--skip_shard", default=0, type=int, help="")
    parser.add_argument("--checkpoint_dir", type=str, default=WORK_DIR+"checkpoint", help="")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def custom_data_collator(batch_samples):
    input_ids = torch.stack([sample['input_ids'] for sample in batch_samples])
    pixel_values = torch.stack([sample['pixel_values'] for sample in batch_samples])
    attention_mask = torch.stack([sample['attention_mask'] for sample in batch_samples])
    labels = torch.stack([sample['labels'] for sample in batch_samples])
    #input_ids.clone().to(input_ids.device)  # Autoregressive
    return (input_ids, pixel_values, attention_mask, labels), labels
    
def main():
    args = set_args()
    if args.local_rank == -1:
        print('Please use `deepspeed *.py` to start')
        return
    else:    
        deepspeed.init_distributed()#dist_backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Deepspeed initialzing on rank {rank} / {world_size} ... {torch.cuda.current_device()}")
        print(f"rank {rank} {torch.cuda.mem_get_info()}")

    model = LlavaForConditionalGeneration.from_pretrained(
        '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/',
        device_map="cpu",
        torch_dtype="auto"
    )
    model.enable_input_require_grads()

    if args.multi_modal_projector_pretraining:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = False
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True
    else:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = True
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True

    model_pipe = PipelineModule(
        layers=get_layer_specs(model), 
        partition_method="type:LanguageModelLayerWrapper",
        num_stages=args.num_stages,
        loss_fn=loss_fn
    )
    print("Rank", rank, "initialized with CUDA_MEM", torch.cuda.mem_get_info(rank))
    
    num_train_epochs = args.num_train_epochs
    num_all_training_steps = math.ceil(args.num_train_epochs * DATASET_SIZE / (args.gradient_accumulation_steps * args.dataloader_batch_size))
    warmup_ratio = 0.03
    num_warmup_steps = warmup_ratio * num_all_training_steps

    ds_config = {
        #"bf16": {"enabled": "auto"},
        "train_batch_size": args.train_batch_size,
        "train_micro_batch_size_per_gpu": args.train_micro_batch_size_per_gpu, #dataloader_batch_size already defined hence train_micro_batch_size_per_gpu is always 1 
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": args.steps_per_print,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": num_all_training_steps,
                "warmup_min_ratio": 0.1,
                "warmup_num_steps": num_warmup_steps,
            }
        },
        "pipeline": {
            "use_reentrant": False
        },        
        "csv_monitor": {
            "enabled": True,
            "output_path": args.checkpoint_dir,
            "job_name": "deepspeed_monitor_logs"
        }
    }
    # we pass optimizer to config, it doesn't matter whether GPUs host any trainable params or not
    # Tensorboard or wandb are not supported on multi-user HPC systems for security reasons
    # https://doc.nhr.fau.de/apps/tensorflow/?h=tensorbo 

    model_pipe_parameters = list((filter(lambda p: p.requires_grad, model_pipe.parameters())))
    model_parameters = list((filter(lambda p: p.requires_grad, model.parameters())))

    engine, _, _, _ = deepspeed.initialize(
        model=model_pipe, 
        model_parameters=model_parameters,
        config=ds_config, 
        #optimizer=None, lr_scheduler=lr_scheduler,
        #training_data=training_data
    )
    # optimizer and lr_scheduler are already passed via deepspeed config
    # instead of provide training_data, we call our own data loader iterator to easily swap bewteen dataset shards
    engine.train()
    print(f"Deepspeed engine initialized at --- RANK {rank} --- hosting {len(model_pipe_parameters)} of {len(model_parameters)} trainable parameters")
    
    if args.skip_shard > 0:
        print_rank_0(f"Loading latest model checkpoint at shard {args.skip_shard}", rank)
        engine.load_checkpoint(load_dir=args.checkpoint_dir)

    for shard_id in tqdm(range(NUM_DATASET_SHARDS)):
        if shard_id < args.skip_shard:
            print_rank_0(f"Shard {shard_id} / {args.skip_shard} skipped", rank)
            continue
        tmp_shard_id = 0
        print_rank_0("Loading dataset from "+WORK_DIR+'LLaVA-Pretrain_processed_dataset/shard_'+str(tmp_shard_id), rank)
        processed_dataset = load_from_disk(WORK_DIR+'LLaVA-Pretrain_processed_dataset/shard_'+str(tmp_shard_id))
        
        num_training_steps = math.floor(len(processed_dataset) / (args.gradient_accumulation_steps * args.dataloader_batch_size))
        print_rank_0(f"Training on {(num_training_steps*args.gradient_accumulation_steps*args.dataloader_batch_size)} of {len(processed_dataset)} sentences.", rank)
        
        training_dataloader = DataLoader(
            processed_dataset,
            batch_size=args.dataloader_batch_size,
            shuffle=False,
            collate_fn=custom_data_collator
        )
        training_dataloader = iter(deepspeed.utils.RepeatingLoader(training_dataloader))

        engine.reset_activation_shape()     # avoid deepspeed pipeline buffer shape mismatch between shards
        for step in tqdm(range(num_training_steps)):
            loss = engine.train_batch(data_iter=training_dataloader)
            torch.cuda.empty_cache()
        
        if (shard_id + 1) % args.save_model_shard == 0:
            start_time = time.time()
            print_rank_0(f"Checkpointing at shard {shard_id + 1}", rank)
            engine.save_checkpoint(save_dir=args.checkpoint_dir)#, exclude_frozen_parameters=True)      
            # checkpointing with frozen parameters excluded makes reloading difficult under the setting of pipeline parallelism 
            # here we sacrifice slightly more disk space in exchange of convenience
            print_rank_0("Checkpoint saved using --- %s seconds ---" % (time.time() - start_time), rank)
    
    print_rank_0("--- FINISHED ---", rank)
    start_time = time.time()
    print_rank_0(f"Checkpointing at shard {shard_id + 1}", rank)
    engine.save_checkpoint(save_dir=args.checkpoint_dir)#, exclude_frozen_parameters=True)
    print_rank_0("Checkpoint saved using --- %s seconds ---" % (time.time() - start_time), rank)

    exit()



if __name__ == "__main__":
    main()



