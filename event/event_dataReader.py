#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理

Author: v_zhouxiaojin@baidu.com
"""

import json
import unicodedata
import torch
import numpy as np
import random
import re
from tqdm import tqdm
from pathlib import Path
from utils import LazyDataset
from transformers import BertTokenizer
from collections import namedtuple
from functools import partial

from torch.utils.data import DataLoader, Dataset

random.seed = 82


def is_whitespace(char):
    """判断是否为空字符"""
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


class CharTokenizer(BertTokenizer):
    """构造字符级tokenizer，添加对空字符的支持"""

    def tokenize(self, text, **kwargs):
        """tokenize by char"""
        token_list = []
        for c in text:
            if c.isupper():
                c = c.lower()
            if c in self.vocab:
                token_list.append(c)
            elif is_whitespace(c):
                token_list.append('[unused1]')
            else:
                token_list.append(self.unk_token)
        return token_list


class DataReader(Dataset):
    """数据构造器"""

    def __init__(self, tokenizer_path, max_len, data_path, event, device='cpu', predict=False):
        self.tokenizer = CharTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.device = device
        self.predict = predict
        self.event = event

        self.build_data(data_path)

    def build_data(self, path):
        with open(path, encoding="utf-8") as f:
            self.datas = [json.loads(line.strip()) for line in f]
        self.len = len(self.datas)

    def build_item(self, instance):
        txt = instance["text"]

        """new"""
        bow = ["导致", "影响", "推动", "带动", "拉动", "使得", '随着']
        keyw = ''
        for w in bow:
            if len(re.findall(w, txt)) > 0:
                if keyw == '':
                    keyw += w
                else:
                    keyw += ',' + w

        input_tokens = [self.cls_token] + self.tokenizer.tokenize(keyw) + \
                       [self.sep_token] + self.tokenizer.tokenize(txt)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        segment_ids = [1] * len(input_ids)
        input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        segment_ids = segment_ids + [0] * (self.max_len - len(segment_ids))

        input_ids = input_ids[:self.max_len]
        segment_ids = segment_ids[:self.max_len]

        segment_ids = np.array(segment_ids)
        input_ids = np.array(input_ids)
        attn_mask = segment_ids

        input_ids = torch.tensor(input_ids, dtype=torch.int64).to(self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.int64).to(self.device)
        attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(self.device)

        if self.predict:
            return input_ids, segment_ids, attn_mask, instance["text_id"]
        else:
            answer = [0] * len(self.event)
            for r in instance["result"]:
                reason_type = r["reason_type"]
                result_type = r["result_type"]
                type = reason_type + "#" + result_type
                label = self.event[type]
                answer[label] = 1

            answer = torch.tensor(answer, dtype=torch.float32).to(self.device)

            return input_ids, segment_ids, attn_mask, answer

    def __getitem__(self, item):
        return self.build_item(self.datas[item])

    def __len__(self):
        return self.len


if __name__ == "__main__":
    pass
