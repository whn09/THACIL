import os
import numpy as np

MODEL_OUTPUT_DIR = '../../dataset/log_preprocess/'
TRAIN_MUID = os.path.join(MODEL_OUTPUT_DIR, 'train_muids_map.txt')
TRAIN_CONTENT_ID = os.path.join(MODEL_OUTPUT_DIR, 'train_content_ids_map.txt')
NPY_PATH = os.path.join(MODEL_OUTPUT_DIR, 'answer.npy')
NUMBER_PATH = os.path.join(MODEL_OUTPUT_DIR, 'number.txt')
FILTER_NPY_PATH = os.path.join(MODEL_OUTPUT_DIR, 'train_cover_image_feature.npy')


def load_train_muid_and_content_id(muids_map_file, content_ids_map_file):
    muids_map = {}
    with open(muids_map_file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            params = line.strip().split('\t')
            muids_map[params[0]] = int(params[1])
    content_ids_map = {}
    with open(content_ids_map_file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            params = line.strip().split('\t')
            content_ids_map[params[0]] = int(params[1])
    return muids_map, content_ids_map


def load_number(filename):
    number_map = {}
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            params = line.strip().split(' ')
            content_id = params[1].split('/')[-1].split('_')[0]
            if content_id in number_map:
                print('content_id:', content_id, 'params[0]:', params[0], 'number_map[content_id]:', number_map[content_id])
            number_map[content_id] = int(params[0])
    return number_map


def filter_funny_npy(filename):
    res_numpy = [[] for _ in range(len(content_ids_map))]
    for content_id, idx in content_ids_map.items():
        if content_id not in number_map:
            res_numpy[idx] = [0 for _ in range(1024)]
        else:
            res_numpy[idx] = feature[number_map[content_id]]
    np.save(filename, res_numpy)


muids_map, content_ids_map = load_train_muid_and_content_id(TRAIN_MUID, TRAIN_CONTENT_ID)
print('muids_map:', len(muids_map), 'content_ids_map:', len(content_ids_map))

feature = np.load(NPY_PATH)
print('feature.shape:', feature.shape)

number_map = load_number(NUMBER_PATH)
print('number_map:', len(number_map))

filter_funny_npy(FILTER_NPY_PATH)
