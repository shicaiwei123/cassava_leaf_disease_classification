'''模型训练相关的函数'''

import numpy as np
import pickle
import cv2
import torch
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import csv
import os
from torchtoolbox.tools import mixup_criterion, mixup_data
from torchvision.datasets import ImageFolder
import sys
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import torch.nn as nn

import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from loss.center_loss import CenterLoss
from loss.class_balanced_loss import CB_loss
import torch.nn.functional as F
from margin.CosineMarginProduct import CosineMarginProduct
from margin.ArcMarginProduct import ArcMarginProduct


def calc_accuracy(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for inputs, labels in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            outputs_batch = model(inputs)
        outputs_full.append(outputs_batch)
        labels_full.append(labels)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        FAR = living_wrong / (living_wrong + living_right)
        FRR = spoofing_wrong / (spoofing_wrong + spoofing_right)
        HTER = (FAR + FRR) / 2

        FAR = float("%.6f" % FAR)
        FRR = float("%.6f" % FRR)
        HTER = float("%.6f" % HTER)
        accuracy = float("%.6f" % accuracy)

        return [accuracy, FAR, FRR, HTER]
    else:
        return [accuracy]


def calc_accuracy_metric(model, classfier, test_loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    classfier_saved = classfier.training
    model.train(False)
    classfier.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        classfier.cuda()
    outputs_full = []
    labels_full = []
    for inputs, labels in tqdm(iter(test_loader), desc="Full forward pass", total=len(test_loader),
                               disable=not verbose):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            feature = model(inputs)
            outputs_batch = classfier(feature)
            # print(labels)
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(labels)
    model.train(mode_saved)
    classfier.train(classfier_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    # print("out:", outputs_full)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        FAR = living_wrong / (living_wrong + living_right)
        FRR = spoofing_wrong / (spoofing_wrong + spoofing_right)
        HTER = (FAR + FRR) / 2

        FAR = float("%.6f" % FAR)
        FRR = float("%.6f" % FRR)
        HTER = float("%.6f" % HTER)
        accuracy = float("%.6f" % accuracy)

        return [accuracy, FAR, FRR, HTER]
    else:
        return [accuracy]


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, args, optimizer, multiplier, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = args.total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                # return self.after_scheduler.get_last_lr()
                return [group['lr'] for group in self.optimizer.param_groups]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def train_metric(feature_model, classifier, train_loader, test_loader, args):
    # 初始化分类器 和对应的参数优化器
    if torch.cuda.is_available():
        feature_model = feature_model.cuda()
        classifier = classifier.cuda()

    optimizer = optim.SGD(
        [{'params': feature_model.parameters()}, {'params': classifier.parameters()}],
        lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    if args.loss == "cosloss":
        print("loss is cosine loss")
        metric_loss_func = CosineMarginProduct(args.feature_dim, args.class_num)
    elif args.loss == "arcloss":
        print("loss is arc loss")
        metric_loss_func = ArcMarginProduct(args.feature_dim, args.class_num)
    else:
        metric_loss_func = None

    if torch.cuda.is_available() and metric_loss_func is not None:
        metric_loss_func = metric_loss_func.cuda()

    cross_loss_func = nn.CrossEntropyLoss()

    # 打印训练参数
    print(args)

    # 初始化,打开定时器,创建保存位置
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    log_dir = args.log_root + '/' + args.name + '.csv'

    # 保存argv参数
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    # 判断是否使用余弦衰减
    if args.lr_step:
        print("lr_step is using")
        step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 60, 90],
                                                        gamma=0.1)
        # # 是否使用学习率预热
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=step_scheduler)

            # 训练初始化
    epoch_num = args.train_epoch  # 训练多少个epoch
    log_interval = args.log_interval  # 每隔多少个batch打印一次状态
    save_interval = args.save_interval  # 每隔多少个epoch 保存一次数据

    batch_num = 0
    log_list = []  # 需要保存的log数据列表

    metric_loss_effect = 0
    metric_loss_cache = 0

    epoch = 0
    accuracy_best = 0

    # 如果是重新训练的过程,那么就读取之前训练的状态
    if args.retrain:
        models_dir = args.model_root + '/' + args.name + '.pt'
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            feature_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # 训练

    while epoch < epoch_num:  # 以epoch为单位进行循环

        metric_loss_all = 0
        cross_loss_all = 0
        train_loss = 0

        # 更新学习率
        if args.lr_step:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                step_scheduler.step(epoch)

        print(epoch, optimizer.param_groups[0]['lr'])

        # 训练
        optimizer.zero_grad()  # 优化器梯度初始化为零,不然会累加之前的梯度
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):
            batch_num += 1

            ones = torch.sparse.torch.eye(args.class_num)
            target_onehot = ones.index_select(0, target)

            if torch.cuda.is_available():
                data, target, target_onehot = data.cuda(), target.cuda(), target_onehot.cuda()

            if args.mixup:
                mixup_alpha = args.mixup_alpha
                inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            feature = feature_model(data)  # 把数据输入网络并得到输出，即进行前向传播

            if args.backbone == "densenet121":
                out = F.relu(feature, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                feature = out
            feature = torch.flatten(feature, 1)
            output = classifier(feature)

            # 根据不同的损失函数求损失

            if metric_loss_func is None:
                if args.loss == "focal":
                    alpha = torch.ones(args.batch_size, dtype=torch.float32) * 0.25

                    metric_loss = focal_loss(target_onehot, output, alpha, 2)
                    metric_loss = torch.tensor(metric_loss)
                elif args.loss == "cb":
                    beta = 0.9999
                    gamma = 2.0
                    samples_per_cls = [1, 2, 2, 13, 2]
                    loss_type = "focal"
                    output = output.cpu()
                    target = target.cpu()
                    metric_loss = CB_loss(target, output, samples_per_cls, args.class_num, loss_type, beta, gamma)
                else:
                    metric_loss = 0
            else:
                metric_loss = metric_loss_func(feature, target)
                metric_loss = cross_loss_func(metric_loss, target)

            train_loss += metric_loss.item()

            metric_loss.backward()  # 反向传播求出输出到每一个节点的梯度
            optimizer.step()  # 根据输出到每一个节点的梯度,优化更新参数```````````````````````````````````````````````````
            optimizer.zero_grad()  # 优化器梯度初始化为零,不然会累加之前的梯度
        log_list.append(float("%.6f" % (train_loss / len(train_loader))))

        # 测试模型

        accurary_result = calc_accuracy_metric(model=feature_model, classfier=classifier, test_loader=test_loader,
                                               hter=False)
        accuracy_test = accurary_result[0]

        log_list = log_list + accurary_result

        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            models_dir = args.model_root + '/' + args.name + '.pth'
            print(models_dir)
            torch.save(feature_model.state_dict(), models_dir)

        print(
            "Epoch {}, train_loss={:.5f}, , accuracy_test={:.5f}  accuracy_best={:.5f}".format(epoch,
                                                                                               train_loss / len(
                                                                                                   train_loader),
                                                                                               accuracy_test,
                                                                                               accuracy_best))

        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": feature_model.state_dict(),

                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        # 保存log
        log_list.append(cross_loss_all / len(train_loader))
        log_list.append(metric_loss_all / len(train_loader))
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_centerloss(feature_model, classifier, optimizer, cost, train_loader, test_loader, args):
    '''

    :param model: 要训练的模型
    margin: 度量函数
    :param cost: 损失函数
    :param optimizer: 优化器
    :param train_loader:  测试数据装载
    :param test_loader:  训练数据装载
    :param args: 配置参数
    :return:
    '''

    centerloss = CenterLoss(args.class_num, args.feature_dim)

    cosloss = CosineMarginProduct(in_feature=args.feature_dim, out_feature=args.class_num)
    cosloss = cosloss.cuda()
    crossntropy = nn.CrossEntropyLoss()
    crossntropy = crossntropy.cuda()

    # 打印训练参数
    print(args)

    # 初始化,打开定时器,创建保存位置
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    log_dir = args.log_root + '/' + args.name + '.csv'

    # 保存argv参数
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        multi_step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                                     np.int(args.train_epoch * 2 / 6),
                                                                                     np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=multi_step_scheduler)

    # 训练初始化
    epoch_num = args.train_epoch  # 训练多少个epoch
    log_interval = args.log_interval  # 每隔多少个batch打印一次状态
    save_interval = args.save_interval  # 每隔多少个epoch 保存一次数据

    batch_num = 0
    log_list = []  # 需要保存的log数据列表

    center_loss_effect = 0
    center_loss_cache = 0

    epoch = 0
    accuracy_best = 0

    # 如果是重新训练的过程,那么就读取之前训练的状态
    if args.retrain:
        models_dir = args.model_root + '/' + args.name + '.pt'
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            feature_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # 训练

    while epoch < epoch_num:  # 以epoch为单位进行循环

        center_loss_all = 0
        loss1_all = 0
        train_loss = 0
        cos_loss_all = 0

        # 更新学习率
        if args.lr_decrease == 'cos':
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)

        elif args.lr_decrease == 'multi_step':
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                multi_step_scheduler.step(epoch=epoch)

        # 训练
        optimizer.zero_grad()  # 优化器梯度初始化为零,不然会累加之前的梯度
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):
            batch_num += 1

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            if args.mixup:
                mixup_alpha = args.mixup_alpha
                inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            feature = feature_model(data)  # 把数据输入网络并得到输出，即进行前向传播

            output = classifier(feature)
            cos_output = cosloss(feature, target)
            cos_loss = crossntropy(cos_output, target)

            if args.mixup:
                loss1 = mixup_criterion(cost, output, labels_a, labels_b, lam)
            else:
                loss1 = cost(output, target)

            center_loss = centerloss(feature, target)

            if center_loss_effect <= 3:
                loss = loss1 + 0.1 * cos_loss + args.proportion * center_loss
            else:
                loss = loss1 + 0.1 *cos_loss
            train_loss += loss.item()
            loss1_all += loss1.item()
            center_loss_all += center_loss.item()
            cos_loss_all += cos_loss.item()
            loss.backward()  # 反向传播求出输出到每一个节点的梯度
            optimizer.step()  # 根据输出到每一个节点的梯度,优化更新参数```````````````````````````````````````````````````
            optimizer.zero_grad()  # 优化器梯度初始化为零,不然会累加之前的梯度
        log_list.append(float("%.6f" % (train_loss / len(train_loader))))

        # 测试模型

        accurary_result = calc_accuracy_metric(model=feature_model, classfier=classifier, test_loader=test_loader,
                                               hter=False)
        accuracy_test = accurary_result[0]

        log_list = log_list + accurary_result

        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            models_dir = args.model_root + '/' + args.name + '.pth'
            print(models_dir)
            torch.save(feature_model.state_dict(), models_dir)

        print(
            "Epoch {}, cross_loss={:.5f}, center_loss={:.5f},cos_loss={:.5f}, accuracy_test={:.5f}  accuracy_best={:.5f}".format(
                epoch,
                loss1_all / len(
                    train_loader),
                cos_loss_all / len(train_loader),
                center_loss_all / len(
                    train_loader),
                accuracy_test,
                accuracy_best))

        if center_loss_cache - center_loss_all / len(train_loader) < 2:
            center_loss_effect += 1
        else:
            center_loss_effect = 0

        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": feature_model.state_dict(),

                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        # 保存log
        log_list.append(loss1_all / len(train_loader))
        log_list.append(center_loss_all / len(train_loader))
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
        center_loss_cache = center_loss_all / len(train_loader)
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        multi_step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                                     np.int(args.train_epoch * 2 / 6),
                                                                                     np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=multi_step_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, (data, target) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            if args.mixup:
                mixup_alpha = args.mixup_alpha
                inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            optimizer.zero_grad()

            output = model(data)

            if args.mixup:
                loss = mixup_criterion(cost, output, labels_a, labels_b, lam)
            else:
                loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy(model, loader=test_loader)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease == 'cos':
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)

        elif args.lr_decrease == 'multi_step':
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                multi_step_scheduler.step(epoch=epoch)

        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def test_base(model, cost, test_loader):
    '''
    :param model: 要测试的带参数模型
    :param cost: 损失函数
    :param test_loader: 测试集
    :return:
    '''

    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    true_wrong = 0
    true_right = 0
    false_wrong = 0
    false_right = 0

    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model(data)  # 数据是按照batch的形式喂入的,然后是这里的输出是全连接层的输出结果
        test_loss += (
            cost(output, target)).item()  # sum up batch 求loss 把所有loss值进行累加.一般分类器会有softmax,但是这里没有是因为集成在这个损失函数之中了
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability #输出的结果虽然会过sofamax转成概率值,但是相对打大小是不变的,所以直接输出取最大值对应的索引就是分类结果
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        compare_result = pred.eq(target.data.view_as(pred)).cpu()  # 判断输出和目标数据是不是一样,然后将比较结果转到cpu上
        # target=target.numpy()
        target = target.cpu()
        compare_result = np.array(compare_result)
        for i in range(len(compare_result)):
            if compare_result[i]:
                if target[i] == 1:
                    true_right += 1
                else:
                    false_right += 1
            else:
                if target[i] == 1:
                    true_wrong += 1
                else:
                    false_wrong += 1

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def deploy_base(model, img, transform):
    '''

    :param model: 模型
    :param img: PIL.Image的对象
    :transform 对应图像处理的操作,和test的一样
    :return:
    '''

    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        img_tensor = img_tensor.cuda()
    result = model(img_tensor)

    if use_cuda:
        result = result.cpu()
    result = result.detach().numpy()
    result = result[0]

    return result


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def sne_analysis(model, data_loader):
    '''
    用来可视化模型输出的特征向量,看看类内和类间的间距
    :param model: 输特征向量的模型,而不是最后分类结果的向量,一般是全连接层的倒数第二层的输出
    :param data_loader: 用来测试的数据的dataloader.推荐batch_size 不要太大.4或者8 较为合适
    :return:
    '''
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # 求特征
            feature = model(data)  # 把数据输入网络并得到输出，即进行前向传播
            feature = torch.flatten(feature, 1)

        if batch_idx == 0:
            feature_arr = np.array(feature)
            target_arr = np.array(target)
        else:
            feature_arr = np.vstack((feature_arr, np.array(feature)))
            target_arr = np.hstack((target_arr, np.array(target)))

        if batch_idx > 100:
            break

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(feature_arr)
    fig = plot_embedding(result, target_arr,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show()
