#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理

Author: v_zhouxiaojin@baidu.com
"""

import json
import copy
import collections
import unicodedata
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import Dataset, LazyDataset
from transformers import BertTokenizer
from collections import namedtuple
from functools import partial
import random

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
            if c in self.vocab:
                token_list.append(c)
            elif is_whitespace(c):
                token_list.append('[unused1]')
            else:
                token_list.append(self.unk_token)
        return token_list


class Example(object):
    """构造Example"""

    def __init__(self,
                 input_id,
                 segment_id,
                 attn_mask,
                 answers=None,
                 ):
        self.input_id = input_id
        self.segment_id = segment_id
        self.attn_mask = attn_mask
        self.answers = answers


class DataReader(object):
    """数据构造器"""

    def __init__(self, tokenizer_path, max_len, event_types_schema):
        self.tokenizer = CharTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.events = {}
        #获得每个event对应的数字编码
        for line in event_types_schema:
            reason_type = line["reason_type"]
            result_type = line["result_type"]
            type = reason_type+"#"+result_type
            if type not in self.events:
                self.events[type] = len(self.events)



    def wrapper(self, instance, predict=False):
        """warpper"""

        context = instance["text"]
        input_tokens =[self.cls_token]+ self.tokenizer.tokenize(context)
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        segment_ids = [1]*len(input_ids)
        input_ids = input_ids+[0]*(self.max_len-len(input_ids))
        segment_ids = segment_ids + [0]*(self.max_len-len(segment_ids))

        input_ids = input_ids[:self.max_len]
        segment_ids = segment_ids[:self.max_len]

        segment_ids = np.array(segment_ids)
        input_ids = np.array(input_ids)
        attn_mask = segment_ids

        if predict:
            return Example(input_ids, segment_ids, attn_mask)
        else:
            answer = [0]*len(self.events)
            for r in instance["result"]:
                reason_type = r["reason_type"]
                result_type = r["result_type"]
                type = reason_type+"#"+result_type
                label = self.events[type]
                answer[label] = 1

            answer = np.array(answer)

            return Example(input_ids,segment_ids,attn_mask,answer)


    def build_dataset(self, data_source, predict=False):
        """构造数据集"""
        instances = []
        if isinstance(data_source, (str, Path)):
            with open(data_source) as f:
                for line in tqdm(f):
                    instance = json.loads(line)
                    instances.append(instance)
        else:
            instances = data_source
        wrapper = partial(self.wrapper, predict=predict)

        return LazyDataset(instances, wrapper)


Batch = namedtuple('Batch', ['input_ids', 'sent_ids', 'attn_masks', 'answers'])


def batcher(device='cpu', status='train'):
    """
    batch构造
    用于DataLoader中的collate_fn
    """

    def numpy_to_tensor(array):
        """numpy转换成torch tensor"""
        return torch.from_numpy(array).to(device)

    def batcher_fn(batch):
        """batcher_fn"""

        batch_input_id = []
        batch_segment_id = []
        batch_attn_mask = []
        batch_answer = []
        for instance in batch:
            batch_input_id.append(instance.input_id)
            batch_segment_id.append(instance.segment_id)
            batch_attn_mask.append(instance.attn_mask)
            if status == 'decode':
                continue
            else:
                batch_answer.append(instance.answers)
        batch_input_id = numpy_to_tensor(np.stack(batch_input_id).astype('int64'))
        batch_sent_id = numpy_to_tensor(np.stack(batch_segment_id).astype('int64'))
        batch_attn_mask = numpy_to_tensor(np.stack(batch_attn_mask).astype('float32'))
        if batch_answer:
            batch_answer = numpy_to_tensor(np.stack(batch_answer).astype('float32'))
        batch_pack = Batch(batch_input_id, batch_sent_id, batch_attn_mask, batch_answer)
        return batch_pack

    return batcher_fn


if __name__ == "__main__":
    process = DataReader(tokenizer_path="/raid/tjc/ccks2021fee/roberta",max_len=120)
    line = {"text_id": "1291633", "text": "铁矿：中长期，今年铁矿供需格局明显改善，巴西矿难及飓风对发运的影响，导致铁矿石全年供应走低", "result": [
        {"reason_type": "台风", "reason_product": "", "reason_region": "巴西", "result_region": "", "result_industry": "",
         "result_type": "供给减少", "reason_industry": "", "result_product": "铁矿石"}]}
    process.wrapper(line)
    pass

