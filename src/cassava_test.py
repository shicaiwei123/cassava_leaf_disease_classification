import numpy as np
import torch
import cv2
import torch.nn as nn
import json
import torchvision.models as tm
from PIL import Image
from cassava_dataloader import cassava_test_transform
from lib.processing_utils import get_file_list


def cassava_single(model, cassava_img, display=False):
    '''
    输入单张人脸照片和模型,返回判断结果
    :param model:
    :param cassava_img:
    :return:
    '''

    predict_dict = {"0": "Cassava Bacterial Blight (CBB)", "1": "Cassava Brown Streak Disease (CBSD)",
                    "2": "Cassava Green Mottle (CGM)", "3": "Cassava Mosaic Disease (CMD)", "4": "Healthy"}

    # cassava_img_rgb = cv2.cvtColor(cassava_img, cv2.COLOR_BGR2RGB)
    #
    # # 预测
    cassava_img_pil = Image.open(path)

    cassava_tensor = cassava_test_transform(cassava_img_pil)
    cassava_tensor = torch.unsqueeze(cassava_tensor, 0)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        cassava_tensor = cassava_tensor.cuda()
    result = model(cassava_tensor)

    if use_cuda:
        result = result.cpu()
    result = result.detach()

    if display:
        print(result)

    max_index = torch.max(result, dim=1)[1]
    max_index = max_index.numpy()[0]
    cassava_class = predict_dict[str(max_index)]

    print(cassava_class)


if __name__ == '__main__':
    # path = "/home/shicaiwei/data/cassava/train_data/1/2873149227.jpg"
    img_dir = "/home/shicaiwei/data/cassava/val_data_iid/4"
    img_path_list = get_file_list(img_dir)
    for path in img_path_list:
        img = cv2.imread(path)
        model = tm.inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(2048, 5, bias=True)
        pretrain_state = torch.load(
            "/home/shicaiwei/project/cassava_leaf_disease_classification/output/models/tb_inception_v3_cb_pretrain_True_freeze_False.pth",
            map_location='cpu')
        model.load_state_dict(pretrain_state)

        cassava_single(model=model, cassava_img=path,display=True)
