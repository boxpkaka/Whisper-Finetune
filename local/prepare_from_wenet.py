import json
import os
import sys
import copy
import soundfile
import opencc
import multiprocessing
from tqdm import tqdm
from typing import List

INIT_ITEM = {'audio': {'path': ''},
             'sentence': '',
             'language': '',
             'sentences': [],
             'duration': 0}


def read_file(path: str) -> List:
    with open(path, 'r') as f:
        file = f.readlines()
    return [item.strip().split(' ') for item in file]


def list2dict(wav_list: List, text_list: List, language: str, converter: opencc.OpenCC,
              start: int, end: int, output_queue) -> None:
    local_labeled_data_list = []
    for i in range(start, end):
        item = copy.deepcopy(INIT_ITEM)
        if len(text_list[i]) <= 1:
            continue
        path = wav_list[i][1]
        sentence = text_list[i][1]
        if converter is not None:
            sentence = converter.convert(sentence)
        wav, sr = soundfile.read(path)
        duration = round(wav.shape[0] / sr, 2)
        item['audio']['path'] = path
        item['sentence'] = sentence
        item['language'] = language
        item['sentences'].append({'start': 0, 'end': duration, 'text': sentence})
        item['duration'] = duration
        local_labeled_data_list.append(item)

    output_queue.put(local_labeled_data_list)


def merge_generate_json(path: List, export_path: str, language: str, converter: opencc.OpenCC) -> None:
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
            process = multiprocessing.Process(target=list2dict, args=(wav_list,
                                                                      text_list,
                                                                      language,
                                                                      converter,
                                                                      start,
                                                                      end,
                                                                      output_queue))
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
            data = json.dumps(item, ensure_ascii=False)
            f.write(data + '\n')


if __name__ == "__main__":
    wenet_dir = sys.argv[1]
    save_dir = sys.argv[2]
    language = sys.argv[3]
    character_type = sys.argv[4]

    path_list = [wenet_dir]
    export_root_path = save_dir
    os.makedirs(export_root_path, exist_ok=True)

    if character_type == 'yue':
        print('Convert Simplified to Traditional Chinese')
        converter = opencc.OpenCC('s2t.json')
    elif character_type == 'zh':
        converter = opencc.OpenCC('t2s.json')
        print('Convert Traditional to Simplified Chinese')
    else:
        converter = None

    merge_generate_json(path_list, os.path.join(export_root_path, 'train.json'), language, converter)
