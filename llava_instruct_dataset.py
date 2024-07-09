
TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

from datasets import load_dataset
from tqdm import tqdm


coco_2014_dataset = load_dataset(
    "lmms-lab/COCO-Caption",
    cache_dir=DATASETS_CACHE_DIR
)



val = coco_2014_dataset['val'].select_columns(['question_id', 'image'])



for item in tqdm(val):
    image_id = int(item['question_id'].split('_')[-1].split('.')[0])
    item['image_id'] = image_id

val = val.sort('image_id')


val.filter(lambda example: example['question_id'] == 'COCO_val2014_000000215677.jpg')




test = coco_2014_dataset.select_columns(['question_id', 'image', 'question']) #[20388:]





dataset = load_dataset(
    "liuhaotian/LLaVA-Instruct-150K",
    cache_dir=DATASETS_CACHE_DIR
)