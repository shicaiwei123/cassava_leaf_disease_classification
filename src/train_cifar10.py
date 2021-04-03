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
from imbalance_cifar10_dataloader import imbalance_cifar10_loader
from efficientnet_pytorch import EfficientNet
import models.efficient_densenet as ed
from loss.class_balanced_loss import CB_loss
from loss.bce_balanceed_loss import BCE_balance_Loss
from loss.cbbcebloss import CBBCEB_loss
from loss.focalloss import FocalLoss
from model.resnet_cifar10 import resnet32
import torch


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def imbalance_cifar10_main(args):
    train_loader = imbalance_cifar10_loader(args, train=True)
    test_loader = imbalance_cifar10_loader(args, train=False)

    args.log_name = args.name + '.csv'
    args.model_name = args.name

    #
    model = resnet32(num_classes=args.class_num)
    if args.freeze:
        for p in model.parameters():
            p.requires_grad = False

    if args.loss == 'focal':
        criterion = FocalLoss(class_num=args.class_num, alpha=[5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50])
    elif args.loss == 'cbloss':
        samples_per_cls = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
        loss_type = "focal"
        criterion = CB_loss(samples_per_cls=samples_per_cls, class_num=args.class_num, loss_type=loss_type)
    elif args.loss == "bcebloss":
        samples_per_cls = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
        criterion = BCE_balance_Loss(class_num=args.class_num, alpha=samples_per_cls, size_average=True, beta=args.beta)
    elif args.loss == 'cbbcebloss':
        samples_per_cls = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
        criterion = CBBCEB_loss(samples_per_cls=samples_per_cls, class_num=args.class_num, beta=args.beta)
    else:
        criterion = nn.CrossEntropyLoss()

    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        criterion = criterion.cuda()
        print("GPU is using")

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
    #                        weight_decay=args.weight_decay)

    args.retrain = False

    train_base(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader,
               args=args)


if __name__ == '__main__':
    seed_torch(2)
    imbalance_cifar10_main(args=args)
