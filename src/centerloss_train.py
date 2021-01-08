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

from lib.model_utils import train_centerloss
from lib.processing_utils import get_file_list, get_mean_std
from cassava_dataloader import cassava_data_loader


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

    model = tm.densenet121(pretrained=True)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    feature_model = model.features
    classifier = nn.Linear(args.feature_dim, args.class_num)

    args.retrain = False

    train_centerloss(feature_model=feature_model, classifier=classifier, train_loader=train_loader,
                     test_loader=test_loader,
                     args=args)


if __name__ == '__main__':
    seed_torch(2)
    centerloss_main(args=args)
