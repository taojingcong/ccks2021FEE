# -*- coding: utf-8 -*-
# @Time : 2021/6/12
# @Software: PyCharm

import argparse
import time
import json
import math
import torch
import torch.nn as nn
import os
import numpy as np

from torch.utils.data import DataLoader
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from argu_data_reader import DataReader
from model import Argument_Extraction
from utils import pickle_load, read_by_line, write_by_line
from pathlib import Path
from copy import deepcopy
from utils import convert_to_numpy


def get_span(logit, threshold=0.5, start_idx=None, end_idx=None):
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = logit.shape[0]
    span = set()
    idx = start_idx
    pos_s = -1
    pos_e = -1
    max_s = threshold
    max_e = threshold
    while idx < end_idx:
        now_s = logit[idx][0]
        now_e = logit[idx][1]
        if pos_s != -1 and pos_e != -1 and now_s >= threshold:
            span.add((pos_s - start_idx, pos_e - start_idx))
            pos_s = pos_e = -1
            max_s = max_e = threshold

        if now_s >= max_s and pos_e == -1:
            max_s = now_s
            pos_s = idx
        if now_e >= max_e and pos_s != -1:
            max_e = now_e
            pos_e = idx
        idx += 1

    if pos_s != -1 and pos_e != -1:
        span.add((pos_s - start_idx, pos_e - start_idx))

    return span


def evaluate(model, dataloader, args, threshold=0.5):
    """"dev函数装饰器"""
    model.eval()
    right_sum = 0
    cnt = 0
    total_ans = 0
    total_pred = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attn_masks, segment_id, answers, pre_len, txt_len = batch
            pre_len = pre_len.numpy()
            txt_len = txt_len.numpy()
            logits = model(input_ids, segment_id, attn_masks)
            for d in range(logits.shape[0]):
                print(cnt, pre_len[d], pre_len[d] + txt_len[d])
                cnt += 1
                pred_set = get_span(logits[d, :, :], threshold=threshold, start_idx=pre_len[d],
                                    end_idx=pre_len[d] + txt_len[d])
                ans_set = get_span(answers[d, :, :], threshold=threshold, start_idx=pre_len[d],
                                   end_idx=pre_len[d] + txt_len[d])

                if args.run_mode == 'Debug':
                    with open(args.save_path / "debug_log.txt", 'a') as f:
                        debug = answers[d, :, :]
                        if int(debug[:, 0].sum().item()) != int(
                                debug[:, 1].sum().item()):
                            print(debug)
                            f.write(f"txt sum not equal {debug} \n")
                        if int(debug[:pre_len[d]].sum().item() + debug[pre_len[d] + txt_len[
                            d]:].sum().item()) != 0:
                            print(debug)
                            f.write(f"pre sum not zero {debug} \n")

                total_pred += len(pred_set)
                total_ans += len(ans_set)
                if len(ans_set) == 0:
                    continue
                tmp_set = ans_set & pred_set
                right_sum += len(tmp_set)

    if total_pred == 0:
        p = 0
    else:
        p = right_sum / total_pred

    r = right_sum / total_ans

    if right_sum == 0:
        f = 0
    else:
        f = 2.0 * (p * r) / (p + r)

    print(right_sum, total_pred, total_ans)
    logs = f'【dev】right: {right_sum}, pred: {total_pred:.6f}, ans: {total_ans:.6f}'
    print(logs)
    with open(args.save_path / "log.txt", 'a') as ff:
        ff.write(logs + '\n')

    model.train()
    return p, r, f


def testPred(args, model, test_dataset: DataReader, threshold=0.5):
    if args.run_mode != 'Quick':
        model.load_state_dict(torch.load(args.save_path / 'best.pth'))
        if args.use_gpu:
            device = args.gpus[0]
        else:
            device = torch.device('cpu')
        model.to(device)
    model.eval()

    p, r, best_f1 = evaluate(model, args.dev_iter, args, args.threshold)
    dev_log = f'evalutation : {p:.6f}, r: {r:.6f}, f1: {best_f1:.6f}'
    print(dev_log)
    with open(args.save_path / 'test_eval.txt', 'a') as f:
        f.write(f"predict eval log: {dev_log} \n")

    pred_ans = {}
    book = {}
    id_set = set()
    type_list = ["reason_region", "reason_product", "reason_industry", "result_region",
                 "result_product", "result_industry"]
    loader = DataLoader(test_dataset, batch_size=1)

    cnt=0
    for batch in loader:
        input_ids, attn_masks, segment_id, id, pre_len, txt_len, ques_type, event_type, txt = batch
        id = id[0]
        txt = txt[0]
        event_type = event_type[0]

        if event_type == '':
            pred_ans[id] = {}
            book[id] = {}
            pred_ans[id]['text_id'] = id
            pred_ans[id]['text'] = txt
            pred_ans[id]['result'] = []
            continue

        pre_len = pre_len.item()
        txt_len = txt_len.item()
        ques_type = ques_type[0]

        with torch.no_grad():
            logits = model(input_ids, segment_id, attn_masks)

        if args.run_mode == 'Debug':
            print(pre_len, pre_len + txt_len)
            with open(args.save_path / 'test_eval.txt', 'a') as f:
                f.write(f"pre_len {pre_len} total_len {pre_len + txt_len} \n")

        print(cnt)
        cnt+=1

        pred_set = get_span(logits[0], threshold=threshold, start_idx=pre_len,
                            end_idx=pre_len + txt_len)
        id_set.add(id)
        if id not in pred_ans.keys():
            pred_ans[id] = {}
            book[id] = {}
            pred_ans[id]['text_id'] = id
            pred_ans[id]['text'] = txt
            pred_ans[id]['result'] = []
        if event_type not in book[id].keys():
            book[id][event_type] = {}
            book[id][event_type]["reason_type"] = event_type.split('#')[0]
            book[id][event_type]["result_type"] = event_type.split('#')[1]
        if ques_type not in book[id][event_type].keys():
            book[id][event_type][ques_type] = ''
        word_set = set()
        for s, e in pred_set:
            word = txt[s:e + 1]
            word_set.add(word)
        for word in word_set:
            if book[id][event_type][ques_type] == '':
                book[id][event_type][ques_type] += word
            else:
                book[id][event_type][ques_type] += ',' + word

    for id in id_set:
        for k in book[id].keys():
            for t in type_list:
                if t not in book[id][k].keys():
                    book[id][k][t] = ''
            pred_ans[id]['result'].append(book[id][k])

    with open(args.save_path / "finnalAns.json", 'w', encoding="utf-8") as w:
        for line in pred_ans:
            w.write(json.dumps(pred_ans[line], ensure_ascii=False) + '\n')


def eva(model, dataloader, threshold):
    model.eval()
    right = 0
    pretotal = 0
    anstotal = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attn_masks, segment_id, answers, pre_len, txt_len = batch
            logits = model(input_ids, segment_id, attn_masks)
            logits = convert_to_numpy(logits)
            answers = convert_to_numpy(answers)
            logits[logits >= threshold] = 1
            logits[logits < threshold] = -1
            right += np.count_nonzero(answers == logits)
            logits[logits < threshold] = 0
            pretotal += np.count_nonzero(logits)
            anstotal += np.count_nonzero(answers)

    if right == 0:
        return 0, 0, 0
    elif pretotal == 0:
        return 0, 0, 0
    else:
        a = right / pretotal
        b = right / anstotal
        c = 2.0 * a * b / (a + b)
        return a, b, c

    model.train()


def train(model, opt, args):
    """train"""

    model.train()
    step = 0
    best_step = -1

    p, r, best_f1 = evaluate(model, args.dev_iter, args, args.threshold)
    print("evalutation done p:{} c:{} f:{}".format(p, r, best_f1))
    model.eval()
    torch.save(model.state_dict(), args.save_path / "best.pth")
    torch.save(opt.state_dict(), args.save_path / 'bestopt.pth')
    model.train()

    # if args.run_mode == 'Debug':
    #     a, b, d = eva(model, args.dev_iter, args.threshold)
    #     print("evalutation done p:{} c:{} f:{}".format(a, b, d))

    loss_fn = nn.BCELoss(reduction="sum")
    for i in range(args.epoch):
        if args.sampler is not None:
            args.sampler.set_epoch(i)
        batch_iter = args.train_iter
        for batch in batch_iter:
            input_ids, attn_masks, segment_id, answers, pre_len, txt_len = batch
            logits = model(input_ids, segment_id, attn_masks)
            step += 1
            loss = loss_fn(logits, answers)

            weight = torch.ones(logits.shape).to(logits.device)
            weight[logits < args.threshold] = args.loss_weight

            loss = loss / logits.shape[0]
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
                role_p, role_r, role_f1 = evaluate(model, args.dev_iter, args, args.threshold)
                dev_log = f'【dev】step: {step}, p: {role_p:.6f}, r: {role_r:.6f}, f1: {role_f1:.6f}, prior best f1: {best_f1:.6f} '
                print(dev_log)
                with open(args.save_path / "log.txt", 'a') as f:
                    f.write(dev_log + '\n')

                # if args.run_mode == 'Debug':
                #     a, b, c = eva(model, args.dev_iter, args.threshold)
                #     dev_log = f'【dev】step: {step}, p: {a:.6f}, r: {b:.6f}, f1: {c:.6f}, prior best f1: {d:.6f} '
                #     with open(args.save_path / "debuglog.txt", 'a') as f:
                #         f.write(dev_log + '\n')
                #     d = max(d, c)

                if role_f1 >= best_f1:
                    best_f1 = role_f1
                    best_step = step
                    model.eval()
                    torch.save(model.state_dict(), args.save_path / "best.pth")
                    torch.save(opt.state_dict(), args.save_path / 'bestopt.pth')
                    model.train()

    model.eval()
    torch.save(model.state_dict(), args.save_path / "last.pth")
    torch.save(opt.state_dict(), args.save_path / 'lastopt.pth')
    model.train()

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

    train_dataset = DataReader(args.encoder_path, args.max_len, args.train_data_path, device, mod=idToEvent, negrad=args.negrad)
    dev_dataset = DataReader(args.encoder_path, args.max_len, args.dev_data_path, device, mod=idToEvent, negrad=args.negrad)
    test_dataset = DataReader(args.encoder_path, args.max_len, args.test_data_path, device, predict=True)

    model = Argument_Extraction(encoder=encoder, input_size=768)

    if args.use_gpu:
        device = args.gpus[0]
    else:
        device = torch.device('cpu')

    if args.load_ckpt != '':
        print('-------load checkpoint-------')
        try:
            model.load_state_dict(torch.load(args.load_ckpt + '/best.pth', map_location=torch.device('cpu')))
            print('-------load model successful-------')
        except:
            print('-------load model failed-------')
        # try:
        #     opt.load_state_dict(torch.load(args.load_ckpt + '/bestopt.pth'))
        #     print('-------load optimizer successful-------')
        # except:
        #     print('-------load optimizer failed-------')

    model.to(device)

    # for state in opt.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print('********init successful********')
    return model, train_dataset, dev_dataset, test_dataset, opt


def main(args):
    model, train_dataset, dev_dataset, test_dataset, opt = init(args)

    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    with open(save_path / "config.json", "w") as w:
        json.dump(args.__dict__, w)
    args.save_path = save_path
    args.sampler = None

    args.train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    args.eval_step = min(args.eval_step, len(args.train_iter))
    total_steps = args.epoch * len(args.train_iter)
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)

    args.scheduler = None
    if args.warmup_ratio > 0:
        args.scheduler = get_linear_schedule_with_warmup(optimizer=opt,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
    args.dev_iter = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    if args.run_mode != 'Quick':
        train(model, opt, args)

    testPred(args, model, test_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MQRC")
    parser.add_argument('--train_data_path', help='Training data path.',
                        default='../data/train.json')
    parser.add_argument('--dev_data_path', help='Dev data path.',
                        default='../data/dev.json')
    parser.add_argument('--test_data_path', help='Test data path.',
                        default='../data/treatment.json')
    parser.add_argument("--events_path", help='event types path',
                        default="../data/reason_result_schema.json")

    parser.add_argument('--encoder_path', help='Pre-train model path.', default='../roberta')

    parser.add_argument('--save_path', help='Checkpoint save path.', default='save')
    parser.add_argument('--load_ckpt', help='Load checkpoint path.', default='2021-07-16_10-45-56')

    parser.add_argument('--loss_weight', help='weight parameter of the predicted label', type=float, default=1)
    parser.add_argument('--max_len', help='Max sequence length.', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--eval_step', type=int, default=10000)
    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--run_mode', type=str, help="Release or Debug", default='Quick')
    parser.add_argument('--negrad', type=int, default=3)
    parser.add_argument('--get', type=int, help="test-0 vaildtion-1 newtest-2", default=0)
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
    # args.get=eval(input('input get'))

    main(args)

    end_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    with open(args.save_path / end_time, 'a') as f:
        f.write('\n')

    if args.get:
        with open(args.save_path / '____DEV____', 'a') as f:
            f.write('\n')
