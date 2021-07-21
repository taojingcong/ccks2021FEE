import argparse
import re
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, get_linear_schedule_with_warmup
from pathlib import Path
from copy import deepcopy
from torch.nn import functional
import numpy as np

parser = argparse.ArgumentParser("MQRC")
parser.add_argument('--train_data_path', help='Training data path.', default='../data/train.json')
parser.add_argument('--dev_data_path', help='Dev data path.', default='../data/dev.json')
parser.add_argument('--test_data_path', help='Test data path.', default='../data/test.json')
parser.add_argument("--events_path", help='event types path', default="data/reason_result_schema.json")
parser.add_argument('--encoder_path', help='Pre-train model path.', default='../roberta')
parser.add_argument('--save_path', help='Checkpoint save path.', default='save')
parser.add_argument('--load_ckpt', help='Load checkpoint path.', default='./save/best.pth')
parser.add_argument('--loss_weight', help='weight parameter of the predicted label', type=float, default=30)
parser.add_argument('--max_len', help='Max sequence length.', type=int, default=240)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eval_step', type=int, default=400)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
parser.add_argument('--use_gpu', type=int, default=1)
args = parser.parse_args()
options = vars(args)


#
# reason_set=set()
# result_set=set()
# dic={}
# event_types_schema = read_by_line(args.events_path)
# for line in event_types_schema:
#     reason_set.add(line["reason_type"])
#     result_set.add(line["result_type"])
#     type=line["reason_type"] + "#" + line["result_type"]
#     if type not in dic.keys():
#         dic[type]=len(dic)
#
# book = np.zeros(len(dic))
# with open("./data/ccks_task2_train.txt", encoding="utf-8") as f:
#     datas = [json.loads(line.strip()) for line in f]
#     for data in datas:
#         result=data["result"]
#         for line in result:
#             type = line["reason_type"] + "#" + line["result_type"]
#             book[dic[type]]+=1
#
# for i in dic:
#     print(i,book[dic[i]])
# print("---------------------------------------")
# for i in reason_set:
#     print(i)
# print("---------------------------------------")
# for i in result_set:
#     print(i)
# pass

# bow = ["导致", "影响","推动","带动","因此","拉动","使得",'随着','由于','因为']
# book={}
# for w in bow:
#     sum = 0
#     num = 0
#     with open("./data/train.json", encoding="utf-8") as f:
#         datas = [json.loads(line.strip()) for line in f]
#         for i in range(len(datas)):
#             txt = datas[i]["text"]
#             l=len(re.findall(w, txt))
#             sum += l
#             if l>0:
#                 num+=1
#                 book[i]=1
#     print(w, sum,num)
#
# print(len(book),len(datas))


def simBCE(pre: torch.Tensor):
    y = np.zeros((pre.shape[0], pre.shape[0]))
    for i in range(pre.shape[0]):
        y[i][i + 1 - i % 2 * 2] = 1
    print(y)
    y = torch.tensor(y,dtype=torch.float32)
    pred = nn.functional.normalize(pre,dim=1)
    pred = torch.matmul(pred, pred.T)
    pred = pred - torch.eye(pred.shape[0]) * 1e12
    pred = torch.sigmoid(pred*20)
    loss_fn = nn.BCELoss(reduction="mean",weight=y)
    loss = loss_fn(pred,y)
    return loss.item()

a=torch.tensor([[1,2,3],[1,2,4]],dtype=torch.float32)
print(simBCE(a))