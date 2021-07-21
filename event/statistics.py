import argparse
import os
import time
import json
import math
import torch
import torch.nn as nn
import pickle
import numpy as np

data_path = {'2021-07-18_22-04-06': 1.1,  # 0.653
             '2021-07-19_09-27-05': 1.8,  # 0.672
             '2021-07-19_11-41-22': 1.4,  # 0.667
             '2021-07-19_13-16-13': 1.4,  # 0.673
             '2021-07-19_16-38-56': 1.4,  # 0.661
             '2021-07-19_17-18-15': 1.3,  # 0.672
             '2021-07-19_21-45-14': 1.4,  # 0.646
             '2021-07-20_00-46-27': 1,  # 0.623
             '2021-07-20_08-30-59': 1.3,  # 0.642
             }

book = {}
threshold = 3.6
total = 0
for path in data_path.keys():
    sum = 0
    w = data_path[path]
    with open('./' + path + '/testB.json', encoding="utf-8") as f:
        datas = [json.loads(line.strip()) for line in f]
    for data in datas:
        if data['text_id'] not in book.keys():
            book[data['text_id']] = {}
            book[data['text_id']]['text'] = data['text']
            book[data['text_id']]['result'] = {}
        for r in data['result']:
            sum += 1
            tp = r['reason_type'] + '#' + r['result_type']
            if tp not in book[data['text_id']]['result'].keys():
                total += 1
                book[data['text_id']]['result'][tp] = w
            else:
                book[data['text_id']]['result'][tp] += w
    print(path, sum)
print('所有(原因#结果)总数', total)

ans = []
num = 0
for i in book.keys():
    d = {'text_id': i, 'text': book[i]['text'], 'result': []}
    for t in book[i]['result']:
        if book[i]['result'][t] >= threshold:
            num += 1
            d['result'].append(dict(reason_type=t.split('#')[0], result_type=t.split('#')[1],
                                    reason_region="", reason_product="", reason_industry="",
                                    result_region="", result_product="", result_industry=""))
    ans.append(d)
print('大于阈值的(原因#结果)总数', num)

with open("../data/treatment.json", 'w', encoding="utf-8") as w:
    for line in ans:
        if line['result'] is not None and len(line['result']):
            w.write(json.dumps(line, ensure_ascii=False) + '\n')
