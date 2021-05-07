import cv2
import numpy as np
from PIL import Image
import time

from lib.processing_utils import get_file_list


def super_green(img, display=False):
    img = np.array(img)
    img1 = np.array(img, dtype='int')  # 转换成int型，不然会导致数据溢出
    # 超绿灰度图
    b, g, r = cv2.split(img1)
    ExG = 2 * g - r - b

    ExG[ExG < 0] = 0
    ExG[ExG > 225] = 225

    ExG = np.array(ExG, dtype='uint8')  # 重新转换成uint8类型
    ret2, th2 = cv2.threshold(ExG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # th2_enhance = green_enhanced(th2)
    #
    # mask = th2_enhance / 255
    # mask = np.uint8(mask)
    # mask_rgb = cv2.merge((mask, mask, mask))
    # img_mask = img * mask_rgb
    #
    #
    # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    # img_mask = Image.fromarray(img_mask)
    # return img_mask

    return th2


def green_enhanced(th2, dispaly=False):
    '''
    对超绿算法进行优化
    :param th2:
    :param dispaly:
    :return:
    '''
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    dilation = cv2.dilate(th2, element1, iterations=1)
    if dispaly:
        cv2.imshow("a", dilation)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    dilation = cv2.dilate(dilation, element2, iterations=1)
    if dispaly:
        cv2.imshow("b", dilation)

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    c_min = []
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if (area < 1000):
            c_min.append(cnt)

    dilation_draw = cv2.drawContours(dilation, c_min, -1, (255, 255, 255), thickness=-1)
    if dispaly:
        cv2.imshow("green_draw", dilation_draw)

    # 4. 膨胀一次，让轮廓突出
    erosion = cv2.erode(dilation_draw, element3, iterations=1)
    if dispaly:
        cv2.imshow("kajh", erosion)

    return erosion

def replaceZeroes(data):
    a=time.time()
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = 1
    b=time.time()
    # print("replace time:",b-a)
    return data



class SSR(object):
    '''
       https://blog.csdn.net/wsp_1138886114/article/details/83096109
       :param src_img:
       :param size:
       :return:
       '''

    def __init__(self, args):
        self.scale = args.ssr_scale
        self.cv2_multi = args.cv2_multi

    def __call__(self, img):

        time_begin=time.time()

        if self.scale[0] < 0:
            return img
        else:
            src_img = np.array(img)
            src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
            weight = 1 / len(self.scale)
            log_R = np.zeros((src_img.shape[0], src_img.shape[1], src_img.shape[2]), dtype=np.float32)
            for size in self.scale:
                size = int(size)
                L_blur = cv2.GaussianBlur(src_img, (size, size), 0)

                src_img_nonzero = np.float32(src_img) + 1
                L_blur = np.float32(L_blur) + 1

                dst_Img = cv2.log(src_img_nonzero / 255.0)
                dst_Lblur = cv2.log(L_blur / 255.0)
                if self.cv2_multi:
                    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
                    log_R += weight * cv2.subtract(dst_Img, dst_IxL)
                else:
                    log_R += weight * cv2.subtract(dst_Img, dst_Lblur)

            dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
            log_uint8 = cv2.convertScaleAbs(dst_R)
            img_arr = cv2.cvtColor(log_uint8, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_arr)

            b = time.time()
            # print("ssr time:", b - time_begin)

            return img_pil


if __name__ == '__main__':

    img_dir = "/home/shicaiwei/data/cassava/train_data/1"
    img_path_list = get_file_list(img_dir)

    for path in img_path_list:
        image = Image.open(path).convert('RGB')
        img = np.array(image)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        cv2.imshow("origin", img)

        ret = super_green(image)
        th2 = green_enhanced(ret, dispaly=True)

        th2_enhance = green_enhanced(th2)

        mask = th2_enhance / 255
        mask = np.uint8(mask)
        mask_rgb = cv2.merge((mask, mask, mask))
        img_mask = img * mask_rgb
        cv2.imshow("img_process", img_mask)

        cv2.waitKey(0)
