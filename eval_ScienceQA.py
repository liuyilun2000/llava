
TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'



import json
import os
from tqdm import tqdm
from PIL import Image

import re
from collections import Counter
import pandas as pd


from datasets import load_dataset


MODEL_NAME = model_name + '_' + checkpoint_name
#MODEL_NAME = 'LLaVA_PEFT_adapter_lora_32_64_' + checkpoint_name
#MODEL_NAME = 'LLaVA_PEFT_lora_32_64'

def remove_special_tokens(input_string, tokens):
    """
    Removes specified special tokens from the input string.
    
    Parameters:
    input_string (str): The string from which to remove special tokens.
    tokens (list): A list of special tokens to be removed.
    
    Returns:
    str: The string with special tokens removed.
    """
    for token in tokens:
        input_string = input_string.replace(token, '')
    return input_string

tokens_to_remove = ['\n\n', '###', '<s>', '</s>']








data = load_dataset(
    "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets/derek-thomas___science_qa",
    split='test'
)



system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

formatting_prompt_prefix = ""

formatting_prompt_infix = "\n"
formatting_prompt_suffix = "\nAnswer directly with the correct option's letter from the given choices only."

results = []

for index, item in enumerate(tqdm(data)):
    image = item['image']
    question = item['question']
    choices = item['choices']
    option_list = ['\nA. ', '\nB. ', '\nC. ', '\nD. ', '\nE. ', '\nF. ']
    option_text = ''
    for index, choice in enumerate(choices):
        option_text += (option_list[index] + choice)
    context = item['hint']
    #
    messages = [
        {"role": "system", "content": system_message}
    ]
    user_messages = [
        formatting_prompt_prefix + ("<image>\n" if image else '') + 'Question: ' + question + formatting_prompt_infix + 'Options: ' + option_text + formatting_prompt_suffix
    ]
    print(user_messages[0])
    answer = generate_conversation(
        additional_prompt="Answer:\n",
        max_new_tokens=64, #add_hist_tokens=True,
        add_generation_prompt=True, 
        msg_hist=messages, msg_queue=user_messages, image=image, verbose=False, output_hist=False)[0]
    #answer = remove_special_tokens(answer, tokens_to_remove)
    result = {
        "index": index,
        'choices': choices,
        "answer": answer
    }
    print(result)
    results.append(result)



output_json_file_path = f'/home/atuin/b207dd/b207dd11/eval/ScienceQA/results_{MODEL_NAME}.json'
with open(output_json_file_path, 'w') as output_file:
    json.dump(results, output_file, indent=4)

print("Results have been written to", output_json_file_path)











MODEL_NAME = model_name + '_' + checkpoint_name

#MODEL_NAME = 'LLaVA_PEFT_lora_128_256'
#MODEL_NAME = 'LLaVA-PEFT_adapter_lora_32_64_save_step1091'
MODEL_NAME = 'LLaVA-PEFT_adapter_32_64_4_top1_save_step1119'
output_json_file_path = f'/home/atuin/b207dd/b207dd11/eval/ScienceQA/results_{MODEL_NAME}.json'

with open(output_json_file_path, 'r') as file:
    results = json.load(file)



def classify_mentions(data):
    classifications = []
    for item in data:
        answer = item.get('answer', '').strip()+'.'
        choices = item.get('choices', [])
        # Match letter options (A., B., C., etc.)
        letter_matches = re.findall(r'\b([A-F])\.', answer)
        letter_counter = Counter(letter_matches)
        #
        total_letter_mentions = sum(letter_counter.values())
        if letter_counter:
            max_letter_mentions = max(letter_counter.values())
            # Check if all letters have the same count and if there are multiple letters
            if len(letter_counter) > 1 and total_letter_mentions == max_letter_mentions * len(choices):
                best_letter = None
            else:
                best_letter = letter_counter.most_common(1)[0][0]
        else:
            max_letter_mentions = 0
            best_letter = None
        # Match choice texts
        text_counter = Counter()
        for choice in choices:
            text_counter[choice] += answer.lower().count(choice.lower())
        #
        total_text_mentions = sum(text_counter.values())
        if total_text_mentions>0 and text_counter:
            max_text_mentions = max(text_counter.values())
            best_text = text_counter.most_common(1)[0][0]
        else:
            max_text_mentions = 0
            best_text = None
        #
        # Determine matched choice
        matched_choice = -1
        if best_letter:
            best_letter_index = ord(best_letter) - 65  
            if best_letter_index < len(choices):
                #if best_text is None or choices[best_letter_index].lower() == best_text.lower():
                matched_choice = best_letter_index
        elif best_text:
            matched_choice = choices.index(best_text)
        if total_letter_mentions==0 and total_text_mentions==0:
            matched_choice = -100
        #
        classifications.append({
            'index': item.get('index'),
            'choices': item.get('choices'),
            'answer': item.get('answer'),
            'total_letter_mentions': total_letter_mentions,
            'max_letter_mentions': max_letter_mentions,
            'best_letter': best_letter,
            'total_text_mentions': total_text_mentions,
            'max_text_mentions': max_text_mentions,
            'best_text': best_text,
            'matched_choice': matched_choice
        })
    return classifications


def manual_assignment(df, num=-1):
    unmatched_df = df[df['matched_choice'] == num]
    unmatched_count = len(unmatched_df)
    count = 1
    for index, row in unmatched_df.iterrows():
        print(f"\n---------- Unmatched instance {count} of {unmatched_count} ----------")
        print(f"Choices: {row['choices']}")
        print(f"Answer: {row['answer']}")
        user_input = input("Please specify the correct choice index (0-based) or press Enter to skip: ").strip()
        if user_input.isdigit() and int(user_input) < len(row['choices']):
            df.at[index, 'matched_choice'] = int(user_input)
            print(user_input, 'assigned.')
        else:
            print(f"Skipped, keeping {num}.")
        count += 1
    return df

classifications = classify_mentions(results)
df = pd.DataFrame(classifications)
# Manually assign values for unmatched choices
df = manual_assignment(df)

output_csv_file = f'/home/atuin/b207dd/b207dd11/eval/ScienceQA/results_{MODEL_NAME}.csv'
df.to_csv(output_csv_file, index=False)







from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



y_true = data['answer']
y_pred = df['matched_choice'].tolist()


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)


print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")


has_image = [(False if img is None else True) for img in data['image']]

# Filter out instances where has_image is False
filtered_y_true = [y for y, img in zip(y_true, has_image) if img]
filtered_y_pred = [p for p, img in zip(y_pred, has_image) if img]

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(filtered_y_true, filtered_y_pred)
precision = precision_score(filtered_y_true, filtered_y_pred, average='macro', zero_division=0)
recall = recall_score(filtered_y_true, filtered_y_pred, average='macro', zero_division=0)
f1 = f1_score(filtered_y_true, filtered_y_pred, average='macro', zero_division=0)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")