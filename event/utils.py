#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
帮助类

Author: v_zhouxiaojin@baidu.com
"""

import json
import pickle
import numpy as np


def convert_to_numpy(tensor):
    """convert to np.ndarray"""
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        return tensor.cpu().numpy()


def pickle_save(obj, path):
    """save by pickle"""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    """load by pickle"""
    with open(path, 'rb') as r:
        return pickle.load(r)


def read_by_line(path):
    """read data by line with json"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def write_by_line(obj, path):
    """write data by line with json"""
    with open(path, 'w', encoding="utf-8") as w:
        for line in obj:
            w.write(json.dumps(line, ensure_ascii=False) + '\n')


class Vocab(object):
    """词表类，封装index to string与 string to index"""

    def __init__(self, itos, stoi):
        """初始化"""
        self.itos = itos
        self.stoi = stoi

    def __len__(self):
        return len(self.itos)


class Dataset(object):
    """Dataset"""

    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, idx):
        """Get the instance with index idx"""
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)


class LazyDataset(object):
    """LazyDataset"""

    def __init__(self, instances, wrapper):
        self.instances = instances
        self.wrapper = wrapper

    def __getitem__(self, idx):
        """Get the instance with index idx"""
        return self.wrapper(self.instances[idx])

    def __len__(self):
        return len(self.instances)

