import argparse
import os
import time
import json
import math
import torch
import torch.nn as nn
import pickle
import numpy as np

result_type = ["reason_region", "reason_product", "reason_industry", "result_region",
               "result_product", "result_industry"]
data_path = {'2021-07-18_23-28-43':1.2,#0.721-147839-2021-07-16_15-53-57--- 1
             '2021-07-19_09-31-34':1.4,#0.816-147630-2021-07-17_23-38-47*-- 1
             '2021-07-19_13-08-16':1.4,#0.810-147643-2021-07-17_23-38-47-- 1
             '2021-07-19_23-47-17':2.5,#0.807-48957-2021-07-19_09-31-34 3
             '2021-07-20_14-44-29':0.5,#0.365-47589-2021-07-14_00-01-36 0
             '2021-07-20_17-05-51':1,#UNK-46254-2021-07-15_21-06-09---- 1
             '2021-07-20_17-06-46':1,#UNK-46509-2021-07-16_10-45-56----- 1
             '2021-07-20_17-10-15':0.5,#0.001-400136-2021-07-18_16-32-00 3
             '2021-07-20_18-04-08':1.6, #0.729-46295-2021-07-16_15-53-57--- 1
             '2021-07-20_18-09-06':2,#0.808-46303-2021-07-17_20-01-56 1
             '2021-07-20_18-10-36':2,#0.806-46103-2021-07-17_23-38-47-- 1
             '2021-07-20_18-12-20':1.8,#0.748-46253-2021-07-16_21-33-02 1
             '2021-07-20_18-52-48':1, #0.655-46212-2021-07-14_21-40-33 0
             '2021-07-20_18-53-55':1, #0.728-46075-2021-07-15_21-06-09---- 1
             '2021-07-20_20-23-40':1, #0.685-46311-2021-07-16_10-45-56----- 1
             }
book = {}
num = 0
total = 0
threshold = 6.5

for path in data_path.keys():
    sum = 0
    wei = data_path[path]
    with open('./' + path + '/finnalAns.json', encoding="utf-8") as f:
        datas = [json.loads(line.strip()) for line in f]
    for data in datas:
        id = data['text_id']
        if id not in book.keys():
            book[id] = {}
            book[id]['text'] = data['text']
            book[id]['result'] = {}
        for r in data['result']:
            t = r['reason_type'] + '#' + r['result_type']
            if t not in book[id]['result'].keys():
                book[id]['result'][t] = {}
                for tp in result_type:
                    book[id]['result'][t][tp] = {}
            for tp in result_type:
                for w in r[tp].split(','):
                    sum += 1
                    if w not in book[id]['result'][t][tp].keys():
                        book[id]['result'][t][tp][w] = wei
                        total += 1
                    else:
                        book[id]['result'][t][tp][w] += wei
    print(path, sum)

result = []
for id in book.keys():
    ans = {'text_id': id, 'text': book[id]['text'], 'result': []}
    for t in book[id]['result'].keys():
        flag = False
        d = {}
        for tp in result_type:
            for w in book[id]['result'][t][tp].keys():
                if book[id]['result'][t][tp][w] >= threshold:
                    num += 1
                    if not flag:
                        d['reason_type'] = t.split('#')[0]
                        d['result_type'] = t.split('#')[1]
                        for k in result_type:
                            d[k] = ''
                        flag = True
                    if d[tp] == '':
                        d[tp] += w
                    else:
                        d[tp] += ',' + w
        if flag:
            ans['result'].append(d)
    result.append(ans)

print("总元素数", total)
print("大于阈值元素数", num)

with open("result.txt", 'w', encoding="utf-8") as f:
    for line in result:
        f.write(json.dumps(line, ensure_ascii=False) + '\n')
