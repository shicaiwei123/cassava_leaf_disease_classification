import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--lr_warmup', type=bool, default=True)
parser.add_argument('--total_epoch', type=int, default=10, help='warmup epoch')
parser.add_argument('--mixup', type=bool, default=False, help='using mixup or not')
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--class_num', type=int, default=10)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--loss', default='bcebloss')
parser.add_argument('--sample_weight', type=bool, default=False)
parser.add_argument('--random_num', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--bias_factor', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.01)

parser.add_argument('--resume_path', type=str, default=None)
parser.add_argument('--pretrain', type=bool, default=True)
parser.add_argument('--freeze', type=bool, default=False)

parser.add_argument('--name', type=str, default='cifar10_resnet32_bceb')
parser.add_argument('--train_dir', type=str,
                    default='/home/data/shicaiwei//utils/cifar10')
parser.add_argument('--test_dir', type=str,
                    default='/home/data/shicaiwei//utils/cifar10')

args_cifar10 = parser.parse_args()

args_cifar10.name = args_cifar10.name + '_pretrain_' + str(args_cifar10.pretrain) + '_freeze_' + str(
    args_cifar10.freeze)

args_cifar10.mixup = False
