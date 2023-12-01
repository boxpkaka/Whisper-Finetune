import json
import random
import os
from tqdm import tqdm
from typing import List, Dict


def get_list(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        data = f.readlines()

    return [json.loads(line) for line in data]


def mix_list(path1: str, path2: str, export_dir) -> None:
    json1 = get_list(path1)
    json2 = get_list(path2)
    mixed_json = json1 + json2
    random.shuffle(mixed_json)

    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, 'train.json'), 'w') as f:
        for line in tqdm(mixed_json):
            data = json.dumps(line)
            f.write(data + '\n')


if __name__ == '__main__':
    path1 = '/data1/yumingdong/whisper/whisper-Finetune/data/cantonese_subset_50/train.json'
    path2 = '/data1/yumingdong/whisper/whisper-Finetune/data/mandarin_50h/train.json'
    export_dir = '/data1/yumingdong/whisper/whisper-Finetune/data/mandarin50h+cantonese50h'

    mix_list(path1, path2, export_dir)


