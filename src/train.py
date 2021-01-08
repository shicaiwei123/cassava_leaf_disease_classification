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


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def resnet50_main(args):
    train_loader = cassava_data_loader(args, train=True)
    test_loader = cassava_data_loader(args, train=False)

    args.log_name = args.name + '.csv'
    args.model_name = args.name

    model = tm.densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, args.class_num, bias=True)
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
