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
from loss.class_balanced_loss import CB_loss

from loss.focalloss import FocalLoss
from lib.model_utils import train_centerloss
import torch


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def centerloss_main(args):
    train_loader = cassava_data_loader(args, train=True)
    test_loader = cassava_data_loader(args, train=False)

    args.log_name = args.name + '.csv'
    args.model_name = args.name

    # get list of models

    # torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
    #
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=args.pretrain)

    feature_model=nn.Sequential(*([m for m in model.children()][:-1]))

    # model = tm.resnet101(pretrained=True)
    # model = tm.vgg16_bn(pretrained=True)
    # model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((7, 7)), Flatten(1))
    if args.freeze:
        for p in model.parameters():
            p.requires_grad = False

    # model.classifier = nn.Linear(4096, args.class_num, bias=True)
    classifier = nn.Linear(args.feature_dim, args.class_num, bias=True)

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

    # 初始化分类器 和对应的参数优化器
    if torch.cuda.is_available():
        feature_model = feature_model.cuda()
        classifier = classifier.cuda()

    optimizer = optim.SGD(
        [{'params': feature_model.parameters()}, {'params': classifier.parameters()}],
        lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    args.retrain = False

    train_centerloss(feature_model=feature_model, classifier=classifier,optimizer=optimizer, cost=criterion,train_loader=train_loader,
                     test_loader=test_loader,
                     args=args)


if __name__ == '__main__':
    seed_torch(2)
    centerloss_main(args=args)
