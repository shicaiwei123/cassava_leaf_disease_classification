import sys

sys.path.append('..')
import shutil
import os
import random
from lib.processing_utils import read_csv


def class_with_csv():
    '''
    利用csv 文件，将train data 分成五类五类保存
    :return:
    '''
    origin_dir = "/home/bbb//shicaiwei/data/cassava/train_images"
    class_dir = "/home/bbb//shicaiwei/data/cassava/train_class"
    csv_path = "/home/bbb//shicaiwei/data/cassava/train.csv"

    csv_readed = read_csv(csv_path)

    for index in range(len(csv_readed)):
        if index == 0:
            continue
        data = csv_readed[index]
        print(data)
        image_name = data[0]
        label = data[1]
        img_path = os.path.join(origin_dir, image_name)
        save_dir = os.path.join(class_dir, label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            shutil.move(img_path, save_dir)
            print(index)
        except Exception as e:
            print(e)


def divide_train_test(num=None, radio=0.2):
    '''
    num  不为 None,每一个类都去num个样本,不然按照radio分
    :param num:
    :param radio:
    :return:
    '''
    # 命名是根据test 和value设置的,iid 就是和训练集分布一样,balance 就是不管训练集分布,只考虑相同
    img_dir = "/home/bbb/shicaiwei/data//cassava/train_class"
    if num is None:
        test_dir = "/home/bbb/shicaiwei/data/cassava/val_data_iid"
        train_dir = "/home/bbb/shicaiwei/data/cassava/train_data_iid"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    else:
        test_dir = "/home/bbb/shicaiwei/data/cassava/val_data_balance"
        train_dir = "/home/bbb/shicaiwei/data/cassava/train_data_balance"

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

    if num != None:
        class_name_list = os.listdir(img_dir)
        for class_index in class_name_list:
            class_dir = os.path.join(img_dir, class_index)

            # for each class
            class_file_list = os.listdir(class_dir)
            random.shuffle(class_file_list)
            test_file_list = class_file_list[:int(num)]
            train_file_List = class_file_list[int(num):]

            for file in test_file_list:
                file_src_path = os.path.join(class_dir, file)
                file_dst_dir = os.path.join(test_dir, class_index)
                if not os.path.exists(file_dst_dir):
                    os.makedirs(file_dst_dir)
                file_dst_path = os.path.join(file_dst_dir, file)
                print(file_src_path)
                shutil.copy(file_src_path, file_dst_path)

            for file in train_file_List:
                file_src_path = os.path.join(class_dir, file)
                file_dst_dir = os.path.join(train_dir, class_index)
                if not os.path.exists(file_dst_dir):
                    os.makedirs(file_dst_dir)
                file_dst_path = os.path.join(file_dst_dir, file)
                shutil.copy(file_src_path, file_dst_path)

                print(file_src_path)
    else:
        class_name_list = os.listdir(img_dir)
        for class_index in class_name_list:
            class_dir = os.path.join(img_dir, class_index)
            class_file_list = os.listdir(class_dir)
            class_file_num = len(class_file_list)
            random.shuffle(class_file_list)
            test_file_list = class_file_list[:int(class_file_num * radio)]
            train_file_List = class_file_list[int(class_file_num * radio):]

            for file in test_file_list:
                file_src_path = os.path.join(class_dir, file)
                file_dst_dir = os.path.join(test_dir, class_index)
                if not os.path.exists(file_dst_dir):
                    os.makedirs(file_dst_dir)
                file_dst_path = os.path.join(file_dst_dir, file)
                shutil.copy(file_src_path, file_dst_path)

                print(file_src_path)

            for file in train_file_List:
                file_src_path = os.path.join(class_dir, file)
                file_dst_dir = os.path.join(train_dir, class_index)
                if not os.path.exists(file_dst_dir):
                    os.makedirs(file_dst_dir)
                file_dst_path = os.path.join(file_dst_dir, file)
                shutil.copy(file_src_path, file_dst_path)

                print(file_src_path)

def divide_train_test_limit():
    '''
    以所有类别中样本数最少的为基准,划分数据,保证数据均衡.
    :return:
    '''

if __name__ == '__main__':
    # class_with_csv()
    divide_train_test(num=200)
