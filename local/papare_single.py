import asyncio
import aiofiles
import json
import os
import copy
import soundfile
from tqdm import tqdm
from typing import List

LANGUAGE = 'yue'
INIT_ITEM = {'audio': {'path': ''},
             'sentence': '',
             'language': LANGUAGE,
             'sentences': [],
             'duration': 0}
COUNT = 0


async def get_file_async(path: str) -> List:
    async with aiofiles.open(path, 'r') as f:
        file = await f.readlines()
    file_list = [item.strip().split(' ') for item in file]
    return file_list


async def list2dict_async(wav_list: List, text_list: List, start: int, end: int) -> List:
    local_labeled_data_list = []
    blank = 0
    for i in range(start, end):
        item = copy.deepcopy(INIT_ITEM)
        if len(text_list[i]) <= 1:
            blank += 1
            continue
        path = wav_list[i][1]
        sentence = text_list[i][1]

        wav, sr = soundfile.read(path)
        duration = round(wav.shape[0] / sr, 2)
        item['audio']['path'] = path
        item['sentence'] = sentence
        item['sentences'].append({'start': 0, 'end': duration, 'text': sentence})
        item['duration'] = duration
        local_labeled_data_list.append(item)

    print('done')
    return local_labeled_data_list


async def generate_json_async(path: str, export_path: str) -> None:
    wav_list = await get_file_async(os.path.join(path, 'wav.scp'))
    text_list = await get_file_async(os.path.join(path, 'text'))

    labeled_data_list = await list2dict_async(wav_list, text_list, 0, len(wav_list))

    print('Writing data ...')
    async with aiofiles.open(export_path, 'w') as f:
        for item in tqdm(labeled_data_list):
            data = json.dumps(item)
            await f.write(data + '\n')


async def merge_generate_json_async(path, export_path):
    num_threads = 16
    labeled_data_list = []

    for split in path:
        wav_list = await get_file_async(os.path.join(split, 'wav.scp'))
        text_list = await get_file_async(os.path.join(split, 'text'))
        length = len(wav_list)
        chunk_size = length // num_threads

        tasks = []
        for i in range(num_threads):
            start = i * chunk_size
            end = length if i == num_threads - 1 else (i + 1) * chunk_size
            task = asyncio.create_task(list2dict_async(wav_list, text_list, start, end))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        for result in results:
            labeled_data_list.extend(result)

    print('Writing data ...')
    async with aiofiles.open(export_path, 'w') as f:
        for item in tqdm(labeled_data_list):
            data = json.dumps(item)
            await f.write(data + '\n')

if __name__ == "__main__":
    path_list = ['/data2/yumingdong/data/cantonese_subset_50/']
    export_root_path = '/data1/yumingdong/whisper/whisper-Finetune/data/cantonese_subset_50'
    os.makedirs(export_root_path, exist_ok=True)

    asyncio.run(merge_generate_json_async(path_list, os.path.join(export_root_path, 'train.json')))
