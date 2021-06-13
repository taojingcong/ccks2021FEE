# -*- coding: utf-8 -*-
# @Time : 2021/6/12
# @Software: PyCharm

#从句子中找到所有的reason type和result type的组合

import argparse
import time
import json
import math
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from event_detection.data_reader import DataReader, batcher
from model import EventDetection
from utils import pickle_load, read_by_line, write_by_line
from pathlib import Path
from copy import deepcopy
from utils import convert_to_numpy
import numpy as np

class Metrix(object):
    def __init__(self,threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.match = 0.
        self.pred_num = 0.
        self.gold_num = 0.

    def update(self,logits,answer):
        "计算每个batch中正确的分类"
        logits = convert_to_numpy(logits)
        answer = convert_to_numpy(answer)

        logits[logits>self.threshold]=1
        match = np.count_nonzero(answer==logits)
        self.match+= match
        self.pred_num+=np.count_nonzero(logits)
        self.gold_num+=np.count_nonzero(answer)

    def calculate(self):
        if self.match==0:
            return 0,0,0

        p = self.match/self.pred_num
        r = self.match/self.gold_num
        f1 = 2.0 * (p * r) / (p + r)

        return p,r,f1

def dev(model,dataloader,threshold):
    """"dev函数装饰器"""

    model.eval()
    metrix = Metrix(threshold)
    with torch.no_grad():
        for batch in dataloader:
            input_ids, segment_ids, attn_masks, answers = batch
            logits = model(input_ids, segment_ids, attn_masks)
            metrix.update(logits,answers)
        p, r, f = metrix.calculate()
    model.train()
    return p, r, f



def train(model, opt, args):
    """train"""
    model.train()

    best_f1 = 0
    step = 0
    best_step = -1
    p,r,best_f1 = dev(model,args.dev_iter,args.threshold)
    loss_fn = nn.BCELoss(reduction='sum')
    for i in range(args.epoch):
        if args.sampler is not None:
            args.sampler.set_epoch(i)
        batch_iter = args.train_iter
        for batch in batch_iter:
            input_ids, segment_id, attn_masks, answers = batch
            logits = model(input_ids, segment_id, attn_masks)
            step += 1
            loss = loss_fn(logits, answers) / logits.shape[0]
            #weight = args.golden_weight*answers+torch.ones(answers.shape).to(answers.device)
            #loss = loss*weight
            loss.backward()
            opt.step()
            model.zero_grad()
            if args.scheduler is not None:
                args.scheduler.step()
            loss_item = loss.item()
            if step % 10 == 0:
                loss_log = f"【train】epoch: {i}, step: {step}, loss: {loss_item: ^7.6f}"
                with open(args.save_path / "loss_log.txt", 'a') as f:
                    f.write(loss_log + '\n')
                print(loss_log)
            if step % args.eval_step == 0:
                role_p, role_r, role_f1 = dev(model,args.dev_iter,args.threshold)
                dev_log = f'【dev】step: {step}, p: {role_p:.6f}, r: {role_r:.6f}, f1: {role_f1:.6f}, prior best f1: {best_f1:.6f} '
                print(dev_log)
                with open(args.save_path / "log.txt", 'a') as f:
                    f.write(dev_log + '\n')

                if role_f1 >= best_f1:
                    best_f1 = role_f1
                    best_step = step
                    if len(args.gpus) > 1:
                        torch.save(model.module.state_dict(), args.save_path / "best.pth")
                    else:
                        torch.save(model.state_dict(), args.save_path / "best.pth")

    if len(args.gpus) > 1:
        torch.save(model.module.state_dict(), args.save_path / "last.pth")
    else:
        torch.save(model.state_dict(), args.save_path / "last.pth")
    best_log = f"Best step is {best_step},  best_f1 is {best_f1}"
    with open(args.save_path / "log.txt", 'a') as f:
        f.write(best_log + '\n')
    print(best_log)


def init(args):
    """初始化模型"""
    print('********init********')

    encoder = BertModel.from_pretrained(args.encoder_path)
    with open(f"{args.encoder_path}/config.json") as f:
        encoder_config = json.load(f)
    args.encoder_dim = encoder_config['hidden_size']

    events_types_schema = read_by_line(args.events_path)
    processor = DataReader(args.encoder_path, args.max_len, events_types_schema)
    model = EventDetection(encoder=encoder,
                           num_event=len(processor.events),
                          input_size=args.encoder_dim)
    if args.load_ckpt != '':
        print('-------load checkpoint-------')
        model.load_state_dict(torch.load(args.load_ckpt, map_location=torch.device('cpu')))
        print('-------load successful-------')
    train_dataset = processor.build_dataset(args.train_data_path)
    dev_dataset = processor.build_dataset(args.dev_data_path)
    test_dataset = None
    if args.test_data_path != '':
        test_dataset = processor.build_dataset(args.test_data_path)
    print('********init successful********')
    return model, train_dataset, dev_dataset, test_dataset, processor


def main(args):
    model, train_dataset, dev_dataset, test_dataset, processor = init(args)
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    with open(save_path / "config.json", "w") as w:
        json.dump(args.__dict__, w)
    args.save_path = save_path
    args.sampler = None
    if args.use_gpu:
        device = args.gpus[0]
    else:
        device = torch.device('cpu')
    model.to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    batch_size = args.batch_size
    args.train_iter = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=batcher(device, 'train'))
    args.eval_step = min(args.eval_step, len(args.train_iter))
    total_steps = args.epoch * len(args.train_iter)
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    args.scheduler = None
    if args.warmup_ratio > 0:
        args.scheduler = get_linear_schedule_with_warmup(optimizer=opt,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
    args.dev_iter = DataLoader(dev_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               collate_fn=batcher(device, 'dev'))

    args.test_iter = None
    if test_dataset is not None:
        args.test_iter = DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=batcher(device, 'dev'))

    train(model, opt, args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser("MQRC")
    parser.add_argument('--train_data_path', help='Training data path.', default='../data/train.json')
    parser.add_argument('--dev_data_path', help='Dev data path.', default='../data/dev.json')
    parser.add_argument('--test_data_path', help='Test data path.', default='')
    parser.add_argument("--events_path",help='event types path',default="../data/reason_result_schema.json")

    parser.add_argument('--encoder_path', help='Pre-train model path.', default='../roberta')

    parser.add_argument('--save_path', help='Checkpoint save path.', default='save')
    parser.add_argument('--load_ckpt', help='Load checkpoint path.', default='')

    parser.add_argument('--golden_weight',help='weight parameter of the golden label',type=float,default=10)
    parser.add_argument('--max_len', help='Max sequence length.', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--eval_step', type=int, default=400)
    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--use_gpu', type=int, default=1)

    args = parser.parse_args()
    options = vars(args)
    print("======================")
    for k, v in options.items():
        print("{}: {}".format(k, v))
    print("======================")

    main(args)