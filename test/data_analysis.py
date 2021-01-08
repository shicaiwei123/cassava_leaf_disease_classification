import cv2

from lib.processing_utils import img_preview, get_file_list

if __name__ == '__main__':
    img_dir = "/home/shicaiwei/data/cassava/train_data"

    file_list = get_file_list(img_dir)
    for file_path in file_list:
        img = cv2.imread(file_path)
        print(img.shape)
        img_resize = cv2.resize(img, (224, 224))
        cv2.imshow("resize", img_resize)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    # img_preview(img_dir=img_dir)
