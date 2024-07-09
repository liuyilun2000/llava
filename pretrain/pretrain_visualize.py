import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
lr_data = pd.read_csv("/home/atuin/b207dd/b207dd11/checkpoint/deepspeed_monitor_logs/Train_Samples_lr.csv")
train_loss_data = pd.read_csv("/home/atuin/b207dd/b207dd11/checkpoint/deepspeed_monitor_logs/Train_Samples_train_loss.csv")

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



fig, ax1 = plt.subplots(figsize=(16, 8))

color = 'tab:red'
ax1.set_xlabel('Step')
ax1.set_ylabel('Learning Rate', color=color)
ax1.plot(lr_data['step'], lr_data['lr'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Train Loss', color=color)  # we already handled the x-label with ax1
ax2.plot(train_loss_data['step'], train_loss_data['train_loss'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.xaxis.set_major_locator(plt.MaxNLocator(20)) 


fig.tight_layout()  # otherwise the right y-label is slightly clipped
file_path = '/home/hpc/b207dd/b207dd11/llava/pretrain_plot.png'
plt.savefig(file_path)


for label in ax1.get_xticklabels():
    label.set_rotation(45)  # Rotate the labels 45 degrees

# Show plot with adjusted x-axis labels
plt.show()