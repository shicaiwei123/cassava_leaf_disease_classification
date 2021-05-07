import torchvision.transforms as tt
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler

from lib.processing_utils import get_mean_std
from lib.process import super_green, SSR
from configuration.config import args

cassava_train_transform = tt.Compose([
    tt.Resize((299, 299)),
    tt.RandomHorizontalFlip(),
    # SSR(args=args),
    tt.ToTensor(),
    tt.Normalize(mean=[0.4293028 , 0.4976464,  0.31065243,],std=[0.22489005, 0.22692077, 0.21530369])
    # tt.Normalize(mean=[0.9272225, 0.9445831, 0.859572], std=[0.08208445, 0.06806655, 0.1649081 ])  #msr
])

cassava_test_transform = tt.Compose([
    tt.Resize((299, 299)),
    # SSR(args=args),
    tt.ToTensor(),
    tt.Normalize(mean=[0.4293028 , 0.4976464,  0.31065243,],std=[0.22489005, 0.22692077, 0.21530369])
    # tt.Normalize(mean=[0.9272225, 0.9445831, 0.859572], std=[0.08208445, 0.06806655, 0.1649081 ])
])


def cassava_data_loader(args, train=True):
    """
    :param train: train or test fold?
    :param batch_size: batch size, int
    :return: data loader

    """

    train_data_set = ImageFolder(args.train_dir, transform=cassava_train_transform)
    test_data_set = ImageFolder(args.test_dir, transform=cassava_test_transform)

    # train_mean, train_std = get_mean_std(train_data_set,0.1)
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
                                                 shuffle=False, num_workers=4, sampler=sampler)
        else:
            loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=4)
    else:

        loader = torch.utils.data.DataLoader(test_data_set, batch_size=32,
                                             shuffle=False, num_workers=4)
    return loader
