#module load cuda/12.1.1
#ds_report
#CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed llava/train_parallel_deepspeed_mixtral_adapter.py --num_stages=4


TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'

#DATASET_SIZE = 558128
#NUM_DATASET_SHARDS = 437

DATASET_SIZE = 665298 * 1.1

NUM_IMAGE_SHARDS = 488
NUM_TEXT_SHARDS = 32
NUM_DATASET_SHARDS = NUM_IMAGE_SHARDS + NUM_TEXT_SHARDS
MAP_BATCH_SIZE = 1280

cuda_mem_exceed_shard_skip_list = [
    76,   158,  182,  242,  293,  363,  418,  421,  664,  752, 
    814,  842,  991,  1266, 1366, 
    1425, 1464, 1574, 1728, 2166, 2441, 2563, 2739, 2854, 
    2894, 3089, 3181, 3395, 3576, 3831, 4300, 4589, 4947, 4950] 


llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"


import random
random.seed(42)

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
from transformers import LlavaConfig, LlavaForConditionalGeneration
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel



from mixtral_modification.configuration_mixtral import MixtralAdapterConfig
from mixtral_modification.modeling_mixtral import MixtralAdapterModel, MixtralAdapterForCausalLM

AutoConfig.register("mixtral-adapter", MixtralAdapterConfig)
AutoModel.register(MixtralAdapterConfig, MixtralAdapterModel)
AutoModelForCausalLM.register(MixtralAdapterConfig, MixtralAdapterForCausalLM)



import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe import ProcessTopology
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import comm as dist

from deepspeed_pipeline_model import *


from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

from peft.tuners.tuners_utils import BaseTuner

from peft.config import PeftConfig
from peft.utils import PeftType




def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

def set_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num_stages", type=int, default=8, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    #
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--multi_modal_projector_pretraining", type=bool, default=True, help="")
    #parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    #
    parser.add_argument("--shared_adapter_type", type=str, default="", help="")
    parser.add_argument("--shared_adapter_num", type=int, default=0, help="")
    parser.add_argument("--lora_r", type=int, default=0, help="")
    parser.add_argument("--lora_alpha", type=int, default=0, help="")
    parser.add_argument("--hidden_dim", type=int, default=0, help="")
    #
    parser.add_argument("--dataloader_batch_size", type=int, default=1, help="")
    parser.add_argument("--train_batch_size", type=int, default=128, help="")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128, help="")
    #
    parser.add_argument("--steps_per_print", default=1, type=int, help="")
    parser.add_argument("--save_model_shard", default=20, type=int, help="")
    parser.add_argument("--skip_shard", default=0, type=int, help="")
    parser.add_argument("--checkpoint_dir", type=str, default=WORK_DIR+"checkpoint", help="")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def custom_data_collator(batch_samples):
    input_ids = torch.stack([sample['input_ids'] for sample in batch_samples])
    attention_mask = torch.stack([sample['attention_mask'] for sample in batch_samples])
    labels = input_ids.clone().to(input_ids.device)  # Autoregressive
    #
    if 'pixel_values' in batch_samples[0].keys():
        pixel_values = torch.stack([sample['pixel_values'] for sample in batch_samples])
        return (input_ids, pixel_values, attention_mask, labels), labels
    else:
        return (input_ids, attention_mask, labels), labels
    

def init_trainable_parameters(model):
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            print(name, module)
            # Initialize LoRA matrices lora_A and lora_B
            module.lora_A.weight = nn.Parameter(torch.randn(module.lora_A.weight.size()) * math.sqrt(2 / module.lora_A.weight.size(0)))
            module.lora_B.weight = nn.Parameter(torch.randn(module.lora_B.weight.size()) * 1e-6)  
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            print(f"Initialized LoRA matrices for {name}")
        if hasattr(module, 'adapter_w1') and hasattr(module, 'adapter_w2'):
            print(name, module)
            module.adapter_w1.weight = nn.Parameter(torch.randn(module.adapter_w1.weight.size()) * math.sqrt(2 / module.adapter_w1.weight.size(0)))
            module.adapter_w2.weight = nn.Parameter(torch.randn(module.adapter_w2.weight.size()) * 1e-6)  
            module.adapter_w1.requires_grad = True
            module.adapter_w2.requires_grad = True
            print(f"Initialized adapter matrices for {name}")



def convert_trainable_parameters(model, trainable_param_names):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if any(substring in name for substring in trainable_param_names):
            #print(name)
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    print(
        f"Convert trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )



def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            #print(name)
    print(
        f"Print trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def main():
    args = set_args()
    if args.local_rank == -1:
        print('Please use `deepspeed *.py` to start')
        return
    else:    
        deepspeed.init_distributed()#dist_backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if args.shared_adapter_type == 'LoRA':
        shared_adapter_args = {'r': args.lora_r, 'lora_alpha': args.lora_alpha, 'lora_dropout': 0.1}
    elif args.shared_adapter_type == "Parallel_Adapter":
        shared_adapter_args = {'hidden_dim': args.hidden_dim, 'hidden_act': 'silu', 'dropout': 0.1}

    mixtral_config = MixtralAdapterConfig(
        vocab_size=32002,
        shared_adapter=True,
        shared_adapter_type=args.shared_adapter_type,
        shared_adapter_num=args.shared_adapter_num,
        shared_adapter_args=shared_adapter_args,
#        shared_routing_adapter=True,
#        shared_routing_adapter_num_experts=4,
#        shared_routing_adapter_num_experts_per_tok=2,
#        shared_routing_adapter_type='LoRA',
#        shared_routing_adapter_args={'r': 32, 'lora_alpha': 32, 'lora_dropout': 0.1},
        output_router_logits=True
    )

    pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'

    llava_config = LlavaConfig.from_pretrained(pretrained_model_dir)
    llava_config.text_config = mixtral_config


    model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_dir,
        config=llava_config,
        device_map="cpu",
        torch_dtype="auto"
    )
    

    torch.cuda.empty_cache()

    model.enable_input_require_grads()
    


    if args.skip_shard==0:
        print(" ######################### trying init params ######################### ")
        init_trainable_parameters(model)
    
    trainable_param_names = ['lora_A', 'lora_B', 'adapter_w1', 'adapter_w2']
    convert_trainable_parameters(model, trainable_param_names)

    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True
    print_trainable_parameters(model)

    model.train()

    '''
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
    '''

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
        "train_micro_batch_size_per_gpu": args.train_micro_batch_size_per_gpu, 
        #dataloader_batch_size already defined hence train_micro_batch_size_per_gpu is always 1 
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": args.steps_per_print,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-5,#1e-3,
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

    
    print(f"Deepspeed engine initializing at --- RANK {rank} --- ...")
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
    print(f"Deepspeed engine successfully initialized at --- RANK {rank} --- hosting {len(model_pipe_parameters)} of {len(model_parameters)} trainable parameters")
    
    if args.skip_shard > 0:
        try:
            print_rank_0(f"Loading latest model checkpoint at shard {args.skip_shard}", rank)
            engine.load_checkpoint(load_dir=args.checkpoint_dir)
        except Exception as e:
            print_rank_0(f"Warning: Unable to load latest checkpoint at {args.checkpoint_dir}. Error: {str(e)}", rank)
        

    NUM_IMAGE_SHARDS = 4880
    NUM_TEXT_SHARDS = 318
    NUM_DATASET_SHARDS = NUM_IMAGE_SHARDS + NUM_TEXT_SHARDS

    image_shard_dir = '/home/atuin/b207dd/b207dd11/LLaVA-PEFT_processed_dataset_with_images'
    text_shard_dir = '/home/atuin/b207dd/b207dd11/LLaVA-PEFT_processed_dataset_plain_texts'
    image_shards = [os.path.join(image_shard_dir, f'shard_{i}') for i in range(NUM_IMAGE_SHARDS)]
    text_shards = [os.path.join(text_shard_dir, f'shard_{i}') for i in range(NUM_TEXT_SHARDS)]

    interval = NUM_IMAGE_SHARDS // NUM_TEXT_SHARDS

    dataset_shards = []
    text_index = 0
    for image_index in range(NUM_IMAGE_SHARDS):
        dataset_shards.append(image_shards[image_index])
        if (image_index + 1) % interval == 0 and text_index < NUM_TEXT_SHARDS:
            dataset_shards.append(text_shards[text_index])
            text_index += 1
    
    for shard_id in tqdm(range(len(dataset_shards))):
        if shard_id < args.skip_shard:
            #print_rank_0(f"Shard {shard_id} / {args.skip_shard} skipped", rank)
            continue
        if shard_id in cuda_mem_exceed_shard_skip_list:
            print_rank_0(f"Shard {shard_id} in {cuda_mem_exceed_shard_skip_list}: file {dataset_shards[shard_id]} skipped to avoid exceeding cuda memory", rank)
            continue
        print_rank_0("Loading dataset from "+dataset_shards[shard_id], rank)
        try:
            processed_dataset = load_from_disk(dataset_shards[shard_id])
            
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
            torch.cuda.empty_cache()
            for step in tqdm(range(num_training_steps)):
                loss = engine.train_batch(data_iter=training_dataloader)
                torch.cuda.empty_cache()
            
            if (shard_id + 1) % args.save_model_shard == 0:
                start_time = time.time()
                print_rank_0(f"Checkpointing at shard {shard_id}", rank)
                engine.save_checkpoint(save_dir=args.checkpoint_dir)#, exclude_frozen_parameters=True)      
                # checkpointing with frozen parameters excluded makes reloading difficult under the setting of pipeline parallelism 
                # here we sacrifice slightly more disk space in exchange of convenience
                print_rank_0("Checkpoint saved using --- %s seconds ---" % (time.time() - start_time), rank)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print_rank_0(f"CUDA out of memory error while processing shard {shard_id}. Skipping this shard.", rank)
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    print_rank_0("--- FINISHED ---", rank)
    start_time = time.time()
    print_rank_0(f"Checkpointing at shard {shard_id}", rank)
    engine.save_checkpoint(save_dir=args.checkpoint_dir)#, exclude_frozen_parameters=True)
    print_rank_0("Checkpoint saved using --- %s seconds ---" % (time.time() - start_time), rank)

    exit()



if __name__ == "__main__":
    main()



