import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_step', type=bool, default=True, help='using cosine learning rate decay or not ')
parser.add_argument('--lr_warmup', type=bool, default=False)
parser.add_argument('--mixup', type=bool, default=False, help='using mixup or not')
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--class_num', type=int, default=5)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--metric', default='arc')
parser.add_argument('--feature_dim', type=int, default=1024)
parser.add_argument('--pretrain', type=bool, default=False)


parser.add_argument('--train_dir', type=str,
                    default='/home/bbb/shicaiwei/data/cassava/train_data')
parser.add_argument('--test_dir', type=str,
                    default='/home/bbb/shicaiwei/data/cassava/val_data')

args_metric = parser.parse_args()

args_metric.name = 'desnet121_' + args_metric.metric + '_0.1_100'

args_metric.mixup = False
