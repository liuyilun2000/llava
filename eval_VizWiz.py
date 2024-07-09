import json
import os
from tqdm import tqdm
from PIL import Image



MODEL_NAME = model_name + '_' + checkpoint_name
#MODEL_NAME = 'LLaVA_PEFT_lora_32_64'
#MODEL_NAME = 'LLaVA_PEFT_lora_128_256'

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

tokens_to_remove = ['\n', '###', '<s>', '</s>']















json_file_path = '/home/atuin/b207dd/b207dd11/eval/VizWiz/Annotations/test.json'

images_base_path = '/home/atuin/b207dd/b207dd11/eval/VizWiz/test/'

with open(json_file_path, 'r') as file:
    data = json.load(file)

system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

formatting_prompt = "\nWhen the provided information is insufficient, respond with 'Unanswerable'. Answer the question using a single word or phrase."

results = []

for item in tqdm(data):
    question = item['question']
    image_file_name = item['image']
    image_path = os.path.join(images_base_path, image_file_name)
    if os.path.exists(image_path):
        with Image.open(image_path) as img:
            messages = [
                {"role": "system", "content": system_message}
            ]
            user_messages = [
                "<image>\n" + 'Question: ' + question + formatting_prompt
            ]
            answer = generate_conversation(
                additional_prompt="Answer: ",
                max_new_tokens=20, #add_hist_tokens=True,
                msg_hist=messages, msg_queue=user_messages, image=img, verbose=False, output_hist=False)[0]
            answer = remove_special_tokens(answer, tokens_to_remove)
            result = {
                "image": image_file_name,
                "answer": answer
            }
            print(result)
            results.append(result)
    else:
        print(f"Image file {image_file_name} not found.")



output_json_file_path = f'/home/atuin/b207dd/b207dd11/eval/VizWiz/results_{MODEL_NAME}.json'
with open(output_json_file_path, 'w') as output_file:
    json.dump(results, output_file, indent=4)

print("Results have been written to", output_json_file_path)