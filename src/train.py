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
from lib.processing_utils import get_file_list, get_mean_std, seed_torch
from cassava_dataloader import cassava_data_loader
from efficientnet_pytorch import EfficientNet
from loss.class_balanced_loss import CB_loss

from loss.focalloss import FocalLoss
from lib.model_arch_utils import Flatten
from models.resnet_cbam import resnet50_cbam
import torch


def resnet50_main(args):
    train_loader = cassava_data_loader(args, train=True)
    test_loader = cassava_data_loader(args, train=False)

    args.log_name = args.name + '.csv'
    args.model_name = args.name

    # get list of models

    # torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
    #
    # model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=args.pretrain)

    model = tm.inception_v3(pretrained=args.pretrain, aux_logits=False)
    # model = tm.vgg16_bn(pretrained=False)
    # model.classifier = nn.Sequential(
    #     nn.Linear(512 * 7 * 7, 4096),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(4096, args.class_num),
    # )
    if args.freeze:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(2048, args.class_num, bias=True)

    if args.loss == 'focal':
        criterion = FocalLoss(class_num=args.class_num, alpha=args.samples_per_cls)
    elif args.loss == 'cbloss':
        samples_per_cls = args.samples_per_cls
        loss_type = "focal"
        criterion = CB_loss(samples_per_cls=samples_per_cls, class_num=args.class_num, loss_type=loss_type)
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
    resnet50_main(args=args)
