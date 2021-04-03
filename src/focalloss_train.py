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

from lib.model_utils import train_metric
from lib.processing_utils import get_file_list, get_mean_std
from cassava_dataloader import cassava_data_loader


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def metric_main(args):
    train_loader = cassava_data_loader(args, train=True)
    test_loader = cassava_data_loader(args, train=False)

    args.log_name = args.name + '.csv'
    args.model_name = args.name

    if args.backbone == 'densenet121':
        model = tm.densenet121(pretrained=args.pretrain)
    elif args.backbone == 'resnet18':
        model = tm.resnet18(pretrained=args.pretrain)
    else:
        model = None
        print("error backbone")
        sys.exit(1)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    if args.backbone !="densenet121":
        feature_model=nn.Sequential(*list(model.children())[:-1])
    else:
        feature_model = model.features
    classifier = nn.Linear(args.feature_dim, args.class_num)

    args.retrain = False

    train_metric(feature_model=feature_model, classifier=classifier, train_loader=train_loader,
                 test_loader=test_loader,
                 args=args)


if __name__ == '__main__':
    seed_torch(2)
    metric_main(args=args)
