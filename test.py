
from PIL import Image
import requests
import torch
import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import copy
import numpy as np


TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"

DATASETS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/datasets"

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"



#llava_model = LlavaForConditionalGeneration.from_pretrained(llava_name, cache_dir=TRANSFORMERS_CACHE_DIR)
bakllava_model = LlavaForConditionalGeneration.from_pretrained(bakllava_name, cache_dir=TRANSFORMERS_CACHE_DIR)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_name, cache_dir=TRANSFORMERS_CACHE_DIR)
#mixtral_model = AutoModelForCausalLM.from_pretrained(mixtral_name, cache_dir=TRANSFORMERS_CACHE_DIR)


bakllava_processor = AutoProcessor.from_pretrained(
    bakllava_name, 
    cache_dir=TRANSFORMERS_CACHE_DIR
)

#mixtral_tokenizer = AutoTokenizer.from_pretrained(mixtral_name, cache_dir=TRANSFORMERS_CACHE_DIR)
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_name, cache_dir=TRANSFORMERS_CACHE_DIR)

E_bakllava_original = copy.deepcopy(bakllava_model.language_model.model.embed_tokens.weight.detach())
E_bakllava = copy.deepcopy(E_bakllava_original[:32000])
#E_mixtral = copy.deepcopy(mixtral_model.model.embed_tokens.weight.detach())
E_mistral = copy.deepcopy(mistral_model.model.embed_tokens.weight.detach())


E_bakllava = E_bakllava.to('cuda')
#E_mixtral = E_mixtral.to('cuda')
E_mistral = E_mistral.to('cuda')
U, S, V = torch.svd(E_bakllava)

# Invert S with a threshold to avoid division by very small numbers
threshold = 1e-10
S_inv = torch.diag(1.0 / S[S > threshold])

# Compute the pseudo-inverse
E_bakllava_pinv = V[:, :S_inv.size(0)].matmul(S_inv).matmul(U.t()[:S_inv.size(0), :])

#T = E_bakllava_pinv @ E_mixtral
T = E_bakllava_pinv @ E_mistral

T = T.to(bakllava_model.device)
transform_layer = torch.nn.Linear(T.shape[0], T.shape[1])
with torch.no_grad():
    transform_layer.weight.copy_(T)
    transform_layer.bias.fill_(0)

class TransformedLlavaMultiModalProjector(torch.nn.Module):
    def __init__(self, original_projector, transform_layer):
        super().__init__()
        self.original_projector = original_projector
        self.transform_layer = transform_layer
    def forward(self, image_features):
        hidden_states = self.original_projector(image_features)
        hidden_states = self.transform_layer(hidden_states)
        return hidden_states

new_model = bakllava_model
original_projector = new_model.multi_modal_projector
new_model.multi_modal_projector = TransformedLlavaMultiModalProjector(original_projector, transform_layer)


new_tokenizer = copy.copy(mistral_tokenizer)
new_language_model = copy.copy(mistral_model)

#new_tokenizer = copy.copy(mixtral_tokenizer)
#new_language_model = copy.copy(mixtral_model)

new_tokenizer.add_tokens(['<image>'])
new_language_model.resize_token_embeddings(len(new_tokenizer))
device = new_language_model.device


new_model.language_model = new_language_model#mixtral_model



#prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
prompt = '<image>Describe the image concisely.'
image_path = "/home/hpc/b207dd/b207dd11/llava/australia.jpg"
image = Image.open(image_path)


inputs = bakllava_processor(text=prompt, images=image, return_tensors="pt")


generate_ids = new_model.generate(**inputs, max_new_tokens=10)
bakllava_processor.batch_decode(generate_ids)#, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]