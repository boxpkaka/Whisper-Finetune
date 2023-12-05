import json
import os
import sys
import multiprocessing
import soundfile
from tqdm import tqdm
from typing import List
import copy


INIT_ITEM = {'audio': {'path': ''},
             'sentence': '',
             'language': '',
             'sentences': [],
             'duration': 0}


def read_file(path: str) -> List:
    with open(path, 'r') as f:
        file = f.readlines()
    return [item.strip().split(' ') for item in file]


def list2dict(wav_list: List, text_list: List, language: str, start: int, end: int, output_queue) -> None:
    local_labeled_data_list = []
    for i in range(start, end):
        item = copy.deepcopy(INIT_ITEM)
        if len(text_list[i]) <= 1:
            continue
        path = wav_list[i][1]
        sentence = text_list[i][1]

        wav, sr = soundfile.read(path)
        duration = round(wav.shape[0] / sr, 2)
        item['audio']['path'] = path
        item['sentence'] = sentence
        item['language'] = language
        item['sentences'].append({'start': 0, 'end': duration, 'text': sentence})
        item['duration'] = duration
        local_labeled_data_list.append(item)

    output_queue.put(local_labeled_data_list)


def merge_generate_json(path: List, export_path: str, language: str) -> None:
    num_processes = multiprocessing.cpu_count()
    output_queue = multiprocessing.Queue()

    processes = []
    for split in path:
        wav_list = read_file(os.path.join(split, 'wav.scp'))
        text_list = read_file(os.path.join(split, 'text'))
        length = len(wav_list)
        chunk_size = length // num_processes

        for i in range(num_processes):
            start = i * chunk_size
            end = length if i == num_processes - 1 else (i + 1) * chunk_size
            process = multiprocessing.Process(target=list2dict, args=(wav_list, text_list, language,
                                                                      start, end, output_queue))
            processes.append(process)
            process.start()

    labeled_data_list = []
    for _ in range(num_processes):
        labeled_data_list.extend(output_queue.get())

    for p in processes:
        p.join()

    print('Writing data ...')
    with open(export_path, 'w') as f:
        for item in tqdm(labeled_data_list):
            data = json.dumps(item)
            f.write(data + '\n')


if __name__ == "__main__":
    wenet_dir = sys.argv[1]
    save_dir = sys.argv[2]
    language = sys.argv[3]

    path_list = [wenet_dir]
    export_root_path = save_dir
    os.makedirs(export_root_path, exist_ok=True)

    merge_generate_json(path_list, os.path.join(export_root_path, 'train.json'), language)
