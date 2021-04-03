import torchvision.transforms as tt
import torch
from dataset.imbalance_cifar10 import ImbalanceCIFAR10
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import WeightedRandomSampler
import os
from lib.processing_utils import get_mean_std, get_dataself_hist
import numpy as np

cassava_train_transform = tt.Compose([
    tt.RandomCrop(32, padding=4),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

cassava_test_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])


def imbalance_cifar10_loader(args, train=True):
    """
    :param train: train or test fold?
    :param batch_size: batch size, int
    :return: data loader

    """
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    train_data_set = ImbalanceCIFAR10(args.train_dir, imb_factor=args.bias_factor, transform=cassava_train_transform,
                                      download=True, train=train,
                                      rand_number=args.random_num)
    test_data_set = CIFAR10(args.test_dir, transform=cassava_test_transform, download=True, train=train)

    # label view

    train_label_List=[]
    for data,label in train_data_set:
        train_label_List.append(label)

    train_label_array=np.array(train_label_List)

    train_label_hist=get_dataself_hist(train_label_array)


    test_label_List=[]
    for data,label in test_data_set:
        test_label_List.append(label)

    test_label_array=np.array(test_label_List)

    test_label_hist=get_dataself_hist(test_label_array)

    # train_mean, train_std = get_mean_std(train_data_set)
    # test_mean, test_std = get_mean_std(test_data_set)
    # print("train_mean", train_mean, "train_std", train_std)
    # print("test_mean", test_mean, "test_std", test_std)

    if train:
        if args.sample_weight:
            print("WeightedRandomSampler is using")
            weights = []
            num_samples = 0
            for data, label in train_data_set:
                num_samples += 1
                if label == 0:
                    weights.append(20)
                elif label == 1 or label == 2 or label == 4:
                    weights.append(10)
                else:
                    weights.append(1.6)
            sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
            loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=32, sampler=sampler)
        else:
            loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=32)
    else:

        loader = torch.utils.data.DataLoader(test_data_set, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)
    return loader
