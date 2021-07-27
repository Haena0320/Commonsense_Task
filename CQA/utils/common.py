import numpy as np
import json, pickle
import os
import csv
import sys
sys.path.append(os.getcwd())

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp)

def load_jsonl(input_file):
    data_list = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            data_list.append(json.loads(line))
    return data_list

