
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'
model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_dir,
    device_map="auto",
    torch_dtype="auto"
)

exit()