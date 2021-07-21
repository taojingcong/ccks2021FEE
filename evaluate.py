import argparse
import time
import json
import math
import torch
import torch.nn as nn
import os
import numpy as np


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

    parser.add_argument('--encoder_path', help='Pre-train model path.',
                        default='../roberta')

    parser.add_argument('--save_path', help='Checkpoint save path.',
                        default='save')
    parser.add_argument('--load_ckpt', help='Load checkpoint path.', default='./2021-07-14_21-40-33')

    parser.add_argument('--loss_weight', help='weight parameter of the predicted label',
                        type=float, default=50)
    parser.add_argument('--max_len', help='Max sequence length.', type=int,
                        default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3*1e-6)
    parser.add_argument('--eval_step', type=int, default=10000)
    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--gpus', nargs='+', type=int, default=[3])
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--run_mode', type=str, help="Release or Debug", default='Debug')
    parser.add_argument('--negrad', type=float, default=1)

    args = parser.parse_args()
    options = vars(args)
    print("======================")
    for k, v in options.items():
        print("{}: {}".format(k, v))
    print("======================")