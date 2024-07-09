import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_loss_plot(checkpoint_dir):
# Read the CSV files
    lr_data = pd.read_csv(f"/home/atuin/b207dd/b207dd11/{checkpoint_dir}/deepspeed_monitor_logs/Train_Samples_lr.csv")
    train_loss_data = pd.read_csv(f"/home/atuin/b207dd/b207dd11/{checkpoint_dir}/deepspeed_monitor_logs/Train_Samples_train_loss.csv")
    # Remove non-numeric rows
    lr_data = lr_data[pd.to_numeric(lr_data['lr'], errors='coerce').notnull()]
    train_loss_data = train_loss_data[pd.to_numeric(train_loss_data['train_loss'], errors='coerce').notnull()]
    # Convert columns to numeric
    lr_data['lr'] = pd.to_numeric(lr_data['lr'])
    train_loss_data['train_loss'] = pd.to_numeric(train_loss_data['train_loss'])
    # Find the last minimum step, which indicates the start of the last training period
    last_min_step = max(lr_data[lr_data['step'] == lr_data['step'].min()]['step'].index)
    # Remove all entries before the last minimum step
    lr_data = lr_data[last_min_step:].reset_index(drop=True)
    train_loss_data = train_loss_data[last_min_step:].reset_index(drop=True)
    #
    fig, ax1 = plt.subplots(figsize=(10, 8))
    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Learning Rate', color=color)
    ax1.plot(lr_data['step'], lr_data['lr'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(20)) 
    #
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Train Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(train_loss_data['step'], train_loss_data['train_loss'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    for label in ax1.get_xticklabels():
        label.set_rotation(45)  # Rotate the labels 45 degrees
    file_path = f'/home/hpc/b207dd/b207dd11/llava/{checkpoint_dir}_plot.png'
    plt.savefig(file_path)
    plt.close()





checkpoint_dir = "LLaVA-PEFT_lora_128_256_checkpoint"
save_loss_plot(checkpoint_dir)

checkpoint_dir = "LLaVA-PEFT_lora_32_64_checkpoint"
save_loss_plot(checkpoint_dir)
#checkpoint_dir = "LLaVA-Pretrain_shard_0_checkpoint"






def save_loss_plot(checkpoint_dirs, save_name, figsize):
    # Read the CSV files for the first directory to plot learning rate
    lr_data = pd.read_csv(f"/home/atuin/b207dd/b207dd11/{checkpoint_dirs[0]}/deepspeed_monitor_logs/Train_Samples_lr.csv")
    lr_data = lr_data[pd.to_numeric(lr_data['lr'], errors='coerce').notnull()]
    lr_data['lr'] = pd.to_numeric(lr_data['lr'])
    last_min_step = max(lr_data[lr_data['step'] == lr_data['step'].min()]['step'].index)
    lr_data = lr_data[last_min_step:].reset_index(drop=True)
    #
    fig, ax1 = plt.subplots(figsize=figsize)
    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Learning Rate', color=color)
    ax1.plot(lr_data['step'], lr_data['lr'], color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(20)) 
    #
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Train Loss', color=color)  # we already handled the x-label with ax1
    #
    for checkpoint_dir in checkpoint_dirs:
        train_loss_data = pd.read_csv(f"/home/atuin/b207dd/b207dd11/{checkpoint_dir}/deepspeed_monitor_logs/Train_Samples_train_loss.csv")
        train_loss_data = train_loss_data[pd.to_numeric(train_loss_data['train_loss'], errors='coerce').notnull()]
        train_loss_data['train_loss'] = pd.to_numeric(train_loss_data['train_loss'])
        train_loss_data = train_loss_data[last_min_step:].reset_index(drop=True)
        ax2.plot(train_loss_data['step'], train_loss_data['train_loss'], label=checkpoint_dir, alpha=0.4)
    #
    ax2.tick_params(axis='y', labelcolor=color)
    #
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    for label in ax1.get_xticklabels():
        label.set_rotation(45)  # Rotate the labels 45 degrees
    #
    ax2.legend(loc='upper right')
    file_path = f'{save_name}.png'
    plt.savefig(file_path)
    plt.close()




# List of checkpoint directories
checkpoint_dirs = [
    #"LLaVA-PEFT_lora_128_256_checkpoint",
    #"LLaVA-PEFT_lora_32_64_checkpoint",
    "LLaVA-PEFT_adapter_lora_32_64_checkpoint",
    "LLaVA-PEFT_adapter_lora_32_64_4_checkpoint",
    "LLaVA-PEFT_adapter_32_64_checkpoint",
    "LLaVA-PEFT_adapter_32_64_4_checkpoint"
]

save_loss_plot(
    checkpoint_dirs, 
    save_name="/home/hpc/b207dd/b207dd11/llava/LLaVA-PEFT_lora_adapters",
    figsize=(6,8)
)



# List of checkpoint directories
checkpoint_dirs = [
    "LLaVA-PEFT_lora_128_256_checkpoint",
    "LLaVA-PEFT_lora_32_64_checkpoint",
    "LLaVA-PEFT_adapter_lora_32_64_checkpoint",
    "LLaVA-PEFT_adapter_lora_32_64_4_checkpoint",
    "LLaVA-PEFT_adapter_32_64_checkpoint",
    "LLaVA-PEFT_adapter_32_64_4_checkpoint"
]

save_loss_plot(
    checkpoint_dirs, 
    save_name="/home/hpc/b207dd/b207dd11/llava/LLaVA-PEFT_lora",
    figsize=(20,8)
)


base_folder = '/home/atuin/b207dd/b207dd11'  # Replace with the actual base folder path
monitor_checkpoints(base_folder)


