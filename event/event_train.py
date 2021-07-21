# -*- coding: utf-8 -*-
# @Time : 2021/6/12
# @Software: PyCharm

# 从句子中找到所有的reason type和result type的组合

import argparse
import os
import time
import json
import math
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from event_dataReader import DataReader
from model import EventDetection
from utils import pickle_load, read_by_line, write_by_line
from pathlib import Path
from copy import deepcopy
from utils import convert_to_numpy
import numpy as np


class Metrix(object):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.match = 0.
        self.pred_num = 0.
        self.gold_num = 0.

    def update(self, logits, answer):
        "计算每个batch中正确的分类"
        logits = convert_to_numpy(logits)
        answer = convert_to_numpy(answer)

        logits[logits >= self.threshold] = 1
        logits[logits < self.threshold] = -1
        match = np.count_nonzero(answer == logits)
        logits[logits < self.threshold] = 0
        self.match += match
        self.pred_num += np.count_nonzero(logits)
        self.gold_num += np.count_nonzero(answer)

    def calculate(self):
        if self.match == 0:
            return 0, 0, 0

        p = self.match / self.pred_num
        r = self.match / self.gold_num
        f1 = 2.0 * (p * r) / (p + r)

        return p, r, f1


def evaluate(model, dataloader, threshold):
    """"dev函数装饰器"""

    model.eval()
    metrix = Metrix(threshold)
    with torch.no_grad():
        for batch in dataloader:
            input_ids, segment_ids, attn_masks, answers = batch
            logits = model(input_ids, segment_ids, attn_masks)
            metrix.update(logits, answers)
        p, r, f = metrix.calculate()
    model.train()
    return p, r, f


def simBCE(pre: torch.Tensor):
    y = np.zeros((pre.shape[0], pre.shape[0]))
    for i in range(pre.shape[0]):
        y[i][i + 1 - i % 2 * 2] = 1
    y = torch.tensor(y, dtype=torch.float32).to(pre.device)
    pred = nn.functional.normalize(pre, dim=1)
    pred = torch.matmul(pred, pred.T)
    pred = pred - torch.eye(pred.shape[0]).to(pred.device) * 1e12
    pred = torch.sigmoid(pred * 20)
    loss_fn = nn.BCELoss(reduction="mean", weight=y)
    loss = loss_fn(pred, y)
    return loss.item()


def train(model, opt, args):
    """train"""
    model.train()
    step = 0
    best_step = -1
    p, r, best_f1 = evaluate(model, args.dev_iter, args.threshold)
    print(p, r, best_f1)
    loss_fn = nn.BCELoss(reduction="none")
    for i in range(args.epoch):
        if args.sampler is not None:
            args.sampler.set_epoch(i)
        batch_iter = args.train_iter
        for batch in batch_iter:
            input_ids, segment_id, attn_masks, answers = batch
            logits, bert_emb_1,bert_emb_2 = model(input_ids, segment_id, attn_masks, mod='ned_emb')

            bert_emb = []
            for i in range(bert_emb_1.shape[0]):
                bert_emb.append(bert_emb_1[i].cpu().detach().numpy())
                bert_emb.append(bert_emb_2[i].cpu().detach().numpy())
            bert_emb = torch.tensor(bert_emb)
            bert_emb = bert_emb.to(bert_emb_1.device)

            step += 1
            loss = loss_fn(logits, answers) + simBCE(bert_emb) * 0.1

            weight = torch.ones(logits.shape).to(logits.device)
            weight[answers >= args.threshold] = args.loss_weight

            loss = (loss * weight).sum() / logits.shape[0]
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
                role_p, role_r, role_f1 = evaluate(model, args.dev_iter, args.threshold)
                dev_log = f'【dev】step: {step}, p: {role_p:.6f}, r: {role_r:.6f}, f1: {role_f1:.6f}, prior best f1: {best_f1:.6f} '
                print(dev_log)
                with open(args.save_path / "log.txt", 'a') as f:
                    f.write(dev_log + '\n')

                if role_f1 >= best_f1:
                    best_f1 = role_f1
                    best_step = step
                    model.eval()
                    torch.save(model.state_dict(), args.save_path / "best.pth")
                    torch.save(opt.state_dict(), args.save_path / "bestopt.pth")
                    model.train()

    model.train()
    torch.save(model.state_dict(), args.save_path / "last.pth")
    torch.save(opt.state_dict(), args.save_path / "bestopt.pth")
    best_log = f"Best step is {best_step},  best_f1 is {best_f1}"

    with open(args.save_path / "log.txt", 'a') as f:
        f.write(best_log + '\n')
    print(best_log)

    return model


def test_eval(args, model, event_book, test_data_path, threshold=0.5, mod='test'):
    print("************** 长度cd " + mod + "************")
    model.load_state_dict(torch.load(args.save_path / "best.pth"))
    if args.use_gpu:
        device = args.gpus[0]
    else:
        device = torch.device('cpu')
    model.to(device)
    model.eval()

    role_p, role_r, role_f1 = evaluate(model, args.dev_iter, args.threshold)
    dev_log = f'p: {role_p:.6f}, r: {role_r:.6f}, f1: {role_f1:.6f}'
    print(dev_log)
    with open(args.save_path / 'test_eval.txt', 'a') as f:
        f.write(f"predict eval log: {dev_log} \n")

    cnt = 0
    id_book = {}
    with open(test_data_path, encoding="utf-8") as f:
        test_data = [json.loads(line.strip()) for line in f]

    for data in test_data:
        id_book[data["text_id"]] = cnt
        cnt += 1

    not_pred = 0
    mk = 0
    with torch.no_grad():
        for batch in args.test_iter:
            input_ids, segment_id, attn_masks, answers = batch
            logits = model(input_ids, segment_id, attn_masks)
            for i in range(logits.shape[0]):
                mk += 1
                print(mk)
                idx = int(answers[i])
                idx = id_book[str(idx)]
                result = []
                flag = True
                for j in range(logits.shape[1]):
                    if logits[i, j] >= threshold:
                        flag = False
                        type = event_book[j].split('#')
                        dic = {}
                        dic["reason_type"] = type[0]
                        dic["result_type"] = type[1]
                        dic["reason_region"] = ''
                        dic["reason_product"] = ''
                        dic["reason_industry"] = ''
                        dic["result_region"] = ''
                        dic["result_product"] = ''
                        dic["result_industry"] = ''
                        result.append(dic)
                test_data[idx]["result"] = result
                if flag:
                    test_data[idx]["result"] = None
                    if args.run_mode == 'Debug':
                        print("{} has no logits greater than threshold".format(idx))
                        not_pred += 1
                        with open(args.save_path / 'test_eval.txt', 'a') as f:
                            f.write("{} has no logits greater than threshold".format(idx))

    # with torch.no_grad():
    #     for batch in args.test_iter:
    #         input_ids, segment_id, attn_masks, answers = batch
    #         logits = model(input_ids, segment_id, attn_masks)
    #         for i in range(logits.shape[0]):
    #             idx = int(answers[i])
    #             idx = id_book[str(idx)]
    #             result = []
    #             flag = True
    #             reason_set = set()
    #             result_set = set()
    #             for j in range(logits.shape[1]):
    #                 if logits[i, j] >= threshold:
    #                     flag = False
    #                     type = event_book[j].split('#')
    #                     reason_set.add(type[0])
    #                     result_set.add(type[1])
    #             for reat in reason_set:
    #                 for rest in result_set:
    #                     dic = {}
    #                     dic["reason_type"] = reat
    #                     dic["result_type"] = rest
    #                     dic["reason_region"] = ''
    #                     dic["reason_product"] = ''
    #                     dic["reason_industry"] = ''
    #                     dic["result_region"] = ''
    #                     dic["result_product"] = ''
    #                     dic["result_industry"] = ''
    #                     result.append(dic)
    #             test_data[idx]["result"] = result
    #             if flag:
    #                 test_data[idx]["result"] = []
    #                 if args.run_mode == 'Debug':
    #                     print("{} has no logits greater than threshold".format(idx))
    #                     not_pred += 1
    #                     with open(args.save_path / 'test_eval.txt', 'a') as f:
    #                         f.write("{} has no logits greater than threshold".format(idx))

    if args.run_mode == 'Debug':
        print(not_pred)
        with open(args.save_path / 'test_eval.txt', 'a') as f:
            f.write(f"not predicted {not_pred}")

    if mod == 'dev':
        with open("../evaluate/ " + args.time + ".json", 'w', encoding="utf-8") as w:
            for line in test_data:
                w.write(json.dumps(line, ensure_ascii=False) + '\n')
    if mod == 'test':
        with open(args.save_path / "treatment.json", 'w', encoding="utf-8") as w:
            for line in test_data:
                w.write(json.dumps(line, ensure_ascii=False) + '\n')

    if mod == 'Best':
        with open(args.save_path / "testB.json", 'w', encoding="utf-8") as w:
            for line in test_data:
                if line['result'] is not None:
                    w.write(json.dumps(line, ensure_ascii=False) + '\n')


def init(args):
    """初始化模型"""
    print('********init********')

    encoder = BertModel.from_pretrained(args.encoder_path)
    with open(f"{args.encoder_path}/config.json") as f:
        encoder_config = json.load(f)
    args.encoder_dim = encoder_config['hidden_size']

    events_types_schema = read_by_line(args.events_path)
    reason_set = set()
    result_set = set()
    events = {}
    idToEvent = []
    for line in events_types_schema:
        reason_set.add(line["reason_type"])
        result_set.add(line["result_type"])
    for reason_type in reason_set:
        for result_type in result_set:
            type = reason_type + "#" + result_type
            events[type] = len(events)
            idToEvent.append(type)

    if args.use_gpu:
        device = args.gpus[0]
    else:
        device = torch.device('cpu')

    train_dataset = DataReader(args.encoder_path, args.max_len, args.train_data_path, events, device)
    dev_dataset = DataReader(args.encoder_path, args.max_len, args.dev_data_path, events, device)
    devtest_dataset = None
    if args.get:
        devtest_dataset = DataReader(args.encoder_path, args.max_len, args.dev_data_path, events, device, predict=True)

    test_dataset = DataReader(args.encoder_path, args.max_len, args.test_data_path, events, device, predict=True)
    Best_dataset = DataReader(args.encoder_path, args.max_len, args.Best_data_path, events, device, predict=True)

    model = EventDetection(encoder=encoder, num_event=len(events), input_size=args.encoder_dim)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_ckpt != '':
        print('-------load checkpoint-------')
        try:
            model.load_state_dict(torch.load(args.load_ckpt + '/best.pth', map_location=torch.device('cpu')))
            print('-------load model successful-------')
        except:
            print('-------load model failed-------')
    model.to(device)

    print('********init successful********')
    return model, train_dataset, dev_dataset, test_dataset, Best_dataset, devtest_dataset, opt, idToEvent


def main(args):
    model, train_dataset, dev_dataset, test_dataset, Best_dataset, devtest_dataset, opt, idToEvent = init(args)

    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    with open(save_path / "config.json", "w") as w:
        json.dump(args.__dict__, w)
    args.save_path = save_path

    args.sampler = None

    batch_size = args.batch_size
    args.train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    args.eval_step = min(args.eval_step, len(args.train_iter))
    total_steps = args.epoch * len(args.train_iter)
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    args.scheduler = None
    if args.warmup_ratio > 0:
        args.scheduler = get_linear_schedule_with_warmup(optimizer=opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    args.dev_iter = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    train(model, opt, args)

    if args.get:
        args.test_iter = DataLoader(devtest_dataset, batch_size=batch_size, shuffle=False)
        test_eval(args, model, idToEvent, args.dev_data_path, mod='dev')

    # args.test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # test_eval(args, model, idToEvent, args.test_data_path, mod='test')

    args.test_iter = DataLoader(Best_dataset, batch_size=batch_size, shuffle=False)
    test_eval(args, model, idToEvent, args.test_data_path, mod='Best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MQRC")
    parser.add_argument('--train_data_path', help='Training data path.', default='../data/train.json')
    parser.add_argument('--dev_data_path', help='Dev data path.', default='../data/dev.json')
    parser.add_argument('--test_data_path', help='Test data path.', default='../data/test.json')
    parser.add_argument('--Best_data_path', help='TestB data path.', default='../data/testBBB.json')

    parser.add_argument("--events_path", help='event types path', default="../data/reason_result_schema.json")
    parser.add_argument('--encoder_path', help='Pre-train model path.', default='../roberta')

    parser.add_argument('--save_path', help='Checkpoint save path.', default='save')
    parser.add_argument('--load_ckpt', help='Load checkpoint path.', default='2021-07-20_19-06-11')
    parser.add_argument('--loss_weight', help='weight parameter of the predicted label', type=float, default=2)
    parser.add_argument('--max_len', help='Max sequence length.', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--eval_step', type=int, default=400)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--gpus', nargs='+', type=int, default=[1])
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--run_mode', type=str, help="Release or Debug", default='Run')
    parser.add_argument('--get', type=int, help="get test", default=1)
    parser.add_argument('--time', help="time", default='')
    parser.add_argument('--decribtion', type=str, help="decribtion you model", default='')

    args = parser.parse_args()
    options = vars(args)

    print("======================")
    for k, v in options.items():
        print("{}: {}".format(k, v))
    print("======================")

    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    args.time = start_time
    # args.decription = input("输入这次训练的描述:")
    args.save_path = start_time
    os.makedirs(args.save_path)

    main(args)

    end_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    with open(args.save_path / end_time, 'a') as f:
        f.write('\n')

    if args.get:
        with open(args.save_path / '____DEV____', 'a') as f:
            f.write('\n')
"""
todo:

"""
