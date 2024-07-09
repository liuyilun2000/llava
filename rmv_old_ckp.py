import os
import re
import time

def get_checkpoint_steps(folder_path):
    steps = []
    for entry in os.scandir(folder_path):
        if entry.is_dir() and re.match(r'global_step\d+', entry.name):
            steps.append(int(re.search(r'\d+', entry.name).group()))
    return sorted(steps)

def delete_oldest_checkpoint(folder_path, steps):
    if len(steps) > 3:
        oldest_step = steps[0]
        oldest_folder = os.path.join(folder_path, f'global_step{oldest_step}')
        print(f"Deleting oldest checkpoint folder {oldest_folder} among : {steps}")
        for root, dirs, files in os.walk(oldest_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(oldest_folder)

def monitor_checkpoints(base_folder):
    while True:
        for folder_name in os.listdir(base_folder):
            folder_path = os.path.join(base_folder, folder_name)
            if os.path.isdir(folder_path):
                steps = get_checkpoint_steps(folder_path)
                delete_oldest_checkpoint(folder_path, steps)
        print("Waiting for new checkpoints...")
        time.sleep(300)  # Check every 300 seconds

base_folder = '/home/atuin/b207dd/b207dd11'  # Replace with the actual base folder path
monitor_checkpoints(base_folder)
