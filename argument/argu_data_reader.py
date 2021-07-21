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
from tqdm import tqdm
from pathlib import Path
from utils import LazyDataset
from transformers import BertTokenizer
from collections import namedtuple
from functools import partial
import random
import re
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

    def __init__(self, tokenizer_path, max_len, data_path, divice='cpu', predict=False):
        self.tokenizer = CharTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.divice = divice
        self.predict = predict
        self.data = preData(data_path, predict)
        self.len = self.data.len

    def ques_build(self, instance):
        id = instance[0]
        type = instance[1]
        txt = instance[2]
        ans_type = instance[3]
        """构造论元"""
        input_tokens = [self.cls_token] + self.tokenizer.tokenize(ans_type) + [self.sep_token] + \
                       self.tokenizer.tokenize(type) + [self.sep_token]
        pre_len = len(input_tokens)
        txt_tokens = self.tokenizer.tokenize(txt)
        txt_len = len(txt_tokens)
        input_tokens = input_tokens + txt_tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        segment_ids = [0] * pre_len + [1] * txt_len + [0] * (self.max_len - pre_len - txt_len)

        input_ids = input_ids[:self.max_len]
        segment_ids = segment_ids[:self.max_len]

        if pre_len > self.max_len:
            print("pre_len>self.max_len")

        txt_len = min(txt_len, self.max_len - pre_len)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).to(self.divice)
        attn_mask = torch.tensor(segment_ids, dtype=torch.int64).to(self.divice)
        segment_ids = torch.tensor(segment_ids, dtype=torch.float32).to(self.divice)

        if self.predict:
            return input_ids, attn_mask, segment_ids, id, pre_len, txt_len, ans_type, type, txt
        else:
            """add answer"""
            answer = np.append(np.zeros((pre_len, 2), dtype=np.int32), instance[4], axis=0)
            if answer.shape[0] < self.max_len:
                answer = np.append(answer,
                                   np.zeros((self.max_len - answer.shape[0], 2), dtype=np.int32),
                                   axis=0)
            else:
                answer = answer[:self.max_len, :]
            answer = torch.tensor(answer, dtype=torch.float32).to(self.divice)
            return input_ids, attn_mask, segment_ids, answer, pre_len, txt_len

    def __getitem__(self, item):
        return self.ques_build(self.data.preDataList[item])

    def __len__(self):
        return self.len


class preData():
    """
    将数据处理,主要内容为preDataList,其格式为:
    [ text_id , 时间类型(原因#结果) , 文本内容 ,
      角色(ans_list中的某一个) , 文本标记(max_len*2) ]
    """

    def __init__(self, path, predict=False):
        self.debug = 0
        self.ans_list = ["reason_region", "reason_product", "reason_industry", "result_region",
                         "result_product",
                         "result_industry"]
        self.preDataList = None
        with open(path, encoding="utf-8") as f:
            datas = [json.loads(line.strip()) for line in f]
        book = {}
        self.preDataList = []
        for data in datas:
            # print(len(self.preDataList))
            id = data["text_id"]
            txt = data["text"]
            if not predict:
                result_total = data["result"]
                book.clear()
                for result in result_total:
                    ult_type = result["result_type"]
                    son_type = result["reason_type"]
                    type = son_type + '#' + ult_type
                    if type in book.keys():
                        book[type] = self.find(id, txt, result, book[type])
                    else:
                        book[type] = self.find(id, txt, result)
                for type in book.keys():
                    for ans in book[type].keys():
                        self.preDataList.append([id, type, txt, ans, book[type][ans]])
            else:
                result_total = data["result"]
                book.clear()
                for result in result_total:
                    ult_type = result["result_type"]
                    son_type = result["reason_type"]
                    type = son_type + '#' + ult_type
                    for ans in self.ans_list:
                        self.preDataList.append([id, type, txt, ans])

        self.len = len(self.preDataList)

    def find(self, id, txt, result, dic=None):
        """
        标记函数,在txt中找到result的位置,且标记到dic中
        """
        if dic is None:
            dic = {}
            for ans_type in self.ans_list:
                dic[ans_type] = np.zeros(shape=(len(txt), 2), dtype=np.int32)

        for ans_type in self.ans_list:
            ans_total = result[ans_type]
            if ans_total == '':
                continue
            for ans in ans_total.split(','):
                for i in re.finditer(ans, txt):

                    if self.debug:
                        flag = False
                        for j in range(i.start(), i.end()):
                            if dic[ans_type][j, 0] + dic[ans_type][j, 1] != 0:
                                if not flag:
                                    print(str(id) + " : multiply word corssing")
                                    print("need    :   " + txt[i.start():i.end()])
                                    self.debug += 1
                                    flag = True
                                print(txt[j])

                    if dic[ans_type][i.start(), 0] == 0 and dic[ans_type][i.end() - 1, 1] == 0:
                        dic[ans_type][i.start(), 0] = 1
                        dic[ans_type][i.end() - 1, 1] = 1

        # print(self.debug)
        return dic


if __name__ == "__main__":

    process = DataReader(tokenizer_path="/raid/tjc/ccks2021fee/roberta", max_len=240)
    train_dataset = process.build_ques_data(path="../data/train.json")
    train_iter = DataLoader(train_dataset,
                            batch_size=32,
                            shuffle=True)
    for batch in train_iter:
        print(type(batch))
    pass
