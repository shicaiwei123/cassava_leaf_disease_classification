'用于解决样本不均衡问题,不均衡训练完成之后,固定feature 提取,重新利用重采样训练分类器'

import sys

sys.path.append('..')
import torch
import torch.nn as nn

import torch.optim as optim
from configuration.config import args
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
import datetime
import random
import torchvision.models as tm

from lib.model_utils import train_base
from lib.processing_utils import get_file_list, get_mean_std
from cassava_dataloader import cassava_data_loader
from efficientnet_pytorch import EfficientNet
import models.efficient_densenet as ed
from loss.class_balanced_loss import CB_loss
from loss.focalloss import FocalLoss
import torch


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def resnet50_main(args):
    # train_loader = cassava_data_loader(args, train=True)
    # test_loader = cassava_data_loader(args, train=False)

    args.log_name = args.name + '.csv'
    args.model_name = args.name

    # get list of models

    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=args.pretrain)
    model.fc = nn.Linear(2048, args.class_num, bias=True)
    if args.freeze:
        for p in model.parameters():
            p.requires_grad = False

    model.load_state_dict(torch.load(args.resume_path))

    # model.classifier = nn.Linear(2048, args.class_num, bias=True)
    model.fc = nn.Linear(2048, args.class_num, bias=True)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
    #                        weight_decay=args.weight_decay)

    args.retrain = False

    train_base(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader,
               args=args)


if __name__ == '__main__':
    seed_torch(2)
    resnet50_main(args=args)
