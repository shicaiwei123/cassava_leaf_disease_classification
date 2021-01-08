import torchvision.transforms as tt
import torch
from torchvision.datasets import ImageFolder


from lib.processing_utils import get_mean_std

cassava_train_transform = tt.Compose([
    tt.Resize((224, 224)),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(mean=[0.43052045, 0.49690652, 0.31396672,],std=[0.22081268, 0.22364931, 0.21199282,])
])

cassava_test_transform = tt.Compose([
    tt.Resize((224, 224)),
    tt.ToTensor(),
    tt.Normalize(mean=[0.43052045, 0.49690652, 0.31396672,],std=[0.22081268, 0.22364931, 0.21199282,])
])


def cassava_data_loader(args, train=True):
    """
    :param train: train or test fold?
    :param batch_size: batch size, int
    :return: data loader

    """

    train_data_set = ImageFolder(args.train_dir, transform=cassava_train_transform)
    test_data_set = ImageFolder(args.test_dir, transform=cassava_test_transform)

    # train_mean, train_std = get_mean_std(train_data_set)
    # test_mean, test_std = get_mean_std(test_data_set)
    # print("train_mean", train_mean, "train_std", train_std)
    # print("test_mean", test_mean, "test_std", test_std)

    if train:
        loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
    else:

        loader = torch.utils.data.DataLoader(test_data_set, batch_size=4,
                                             shuffle=False, num_workers=4)
    return loader