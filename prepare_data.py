import numpy as np
import os
import pydicom
from pydicom import dcmread
from pydicom.data import get_testdata_files
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2
import random
import nibabel as nib


class YOLOv7_data_prepare():
    def __init__(self, condition, cross_validation_number):
        self.path = 'C:/Users/CMUHCH_AIAMIC/Desktop/overall data/ROI data/'
        self.condition = condition
        self.cross_validation_number = cross_validation_number

    # data precessing.
    def apply_window_level_width(self, image, window_center, window_width):
        """
            使用 window level 和 window width 進行 normalization。

            Parameters:
            - image: 影像數據 (2D 或 3D NumPy 數組)
            - window_center: window level
            - window_width: window width

            Returns:
            - normalized_image: 正規化後的影像數據
            """
        min_value = window_center - window_width / 2.0
        max_value = window_center + window_width / 2.0
        # 將影像數據截斷到指定的 window 範圍
        clipped_image = np.clip(image, min_value, max_value)
        # 正規化到 [0, 1] 範圍
        normalized_image = (clipped_image - min_value) / (max_value - min_value)
        return normalized_image

    def cross_validation_data(self):
        """
        Statement:
        5-fold cross validation data 準備

        Parameters:
        - cross_validation_number: 第幾份data

        Returns:
        - train_image:訓練資料
        - train_label：訓練標籤
        - train_bounding_box: 訓練偵測框座標
        - train_filename: 訓練資料病患檔名
        - val_image: 驗證資料
        - val_label: 驗證標籤
        - val_bounding_box: 驗證偵測框座標
        - val_filename: 驗證資料病患檔名
        """
        train_image_path = f'D:/yolov7/datasets/cross{self.cross_validation_number}/images/train/'
        train_label_path = f'D:/yolov7/datasets/cross{self.cross_validation_number}/labels/train/'
        val_image_path = f'D:/yolov7/datasets/cross{self.cross_validation_number}/images/val/'
        val_label_path = f'D:/yolov7/datasets/cross{self.cross_validation_number}/labels/val/'

        path = 'C:/Users/CMUHCH_AIAMIC/Desktop/overall data/ROI data/'
        train_name, val_name = [], []
        for num, filename in enumerate(os.listdir(path)):
            if (self.cross_validation_number - 1) * 60 <= num < (self.cross_validation_number - 1) * 60 + 60:
                val_name.append(int(filename[5: 8]))
            else:
                train_name.append(int(filename[5: 8]))

        train_image, train_label, train_bounding_box, train_filename = [], [], [], []
        val_image, val_label, val_bounding_box, val_filename = [], [], [], []

        train_number, val_number = 1, 1
        for num_filename, filename in enumerate(os.listdir(path)):
            print(filename)
            data = loadmat(path + filename)  # 读取mat文件
            raw_image = data['dwi']
            yoloposition = data['yoloposition']
            for i in range(yoloposition.shape[0]):
                num_image = yoloposition[i][1][0][0]
                if num_image == -1:
                    continue

                try:
                    pos1 = yoloposition[i][0][0][0][0]
                    if pos1[0] != 0:
                        if int(filename[5: 8]) in train_name:
                            image = raw_image[:, :, num_image - 1]
                            concate_image1 = raw_image[:, :, num_image]
                            concate_image2 = raw_image[:, :, num_image - 2]
                            image = self.apply_window_level_width(image, window_center=20, window_width=180)

                            # 新增condition 目的為增加前後文資訊量。
                            concate_image1 = self.apply_window_level_width(concate_image1, window_center=20,
                                                                           window_width=180)
                            concate_image2 = self.apply_window_level_width(concate_image2, window_center=20,
                                                                           window_width=180)
                            image = image.reshape(image.shape[0], image.shape[1], 1)
                            concate_image1 = concate_image1.reshape(concate_image1.shape[0], concate_image2.shape[1], 1)
                            concate_image2 = concate_image2.reshape(concate_image2.shape[0], concate_image2.shape[1], 1)
                            image = np.concatenate((concate_image1, image, concate_image2), axis=-1)

                            #                         train_image.append(image)
                            #                         train_filename.append(filename[0:8])
                            #                         train_bounding_box.append([])
                            #                         train_label.append([])
                            #                         train_bounding_box[-1].append(pos1)
                            #                         train_label[-1].append(0)

                            cv2.imwrite(train_image_path + f'{filename[0:8]}_{train_number}.jpg', image * 255)
                            with open(train_label_path + f'{filename[0:8]}_{train_number}.txt',
                                      'w') as train_label_file:
                                for (categorical, points) in zip([0], [pos1]):
                                    train_label_file.write(str(categorical) + ' ')
                                    for p in points:
                                        train_label_file.write(str(p) + " ")
                                    train_label_file.write('\n')
                            train_number += 1

                        if int(filename[5: 8]) in val_name:
                            image = raw_image[:, :, num_image - 1]
                            concate_image1 = raw_image[:, :, num_image]
                            concate_image2 = raw_image[:, :, num_image - 2]
                            image = self.apply_window_level_width(image, window_center=20, window_width=180)

                            # 新增condition 目的為增加前後文資訊量。
                            concate_image1 = self.apply_window_level_width(concate_image1, window_center=20, window_width=180)
                            concate_image2 = self.apply_window_level_width(concate_image2, window_center=20, window_width=180)
                            image = image.reshape(image.shape[0], image.shape[1], 1)
                            concate_image1 = concate_image1.reshape(concate_image1.shape[0], concate_image2.shape[1], 1)
                            concate_image2 = concate_image2.reshape(concate_image2.shape[0], concate_image2.shape[1], 1)
                            image = np.concatenate((concate_image1, image, concate_image2), axis=-1)

                            #                         val_image.append(image)
                            #                         val_filename.append(filename[0:8])
                            #                         val_bounding_box.append([])
                            #                         val_label.append([])
                            #                         val_bounding_box[-1].append(pos1)
                            #                         val_label[-1].append(0)

                            cv2.imwrite(val_image_path + f'{filename[0:8]}_{val_number}.jpg', image * 255)
                            with open(val_label_path + f'{filename[0:8]}_{val_number}.txt', 'w') as val_label_file:
                                for (categorical, points) in zip([0], [pos1]):
                                    val_label_file.write(str(categorical) + ' ')
                                    for p in points:
                                        val_label_file.write(str(p) + " ")
                                    val_label_file.write('\n')
                            val_number += 1

                except:
                    pass

        #     train_image, train_label, train_bounding_box, train_filename = np.array(train_image), np.array(train_label), np.array(train_bounding_box), np.array(train_filename)
        #     val_image, val_label, val_bounding_box, val_filename = np.array(val_image), np.array(val_label), np.array(val_bounding_box), np.array(val_filename)
        #     print(train_image.shape, train_label.shape, train_bounding_box.shape, train_filename.shape)
        #     print(val_image.shape, val_label.shape, val_bounding_box.shape, val_filename.shape)
        #     print(train_filename)
        #     train_data = list(zip(train_image, train_label, train_bounding_box, train_filename))
        #     np.random.shuffle(train_data)
        #     train_data = list(zip(*train_data))
        #     train_image, train_label, train_bounding_box, train_filename = np.array(train_data[0]), np.array(train_data[1]), np.array(train_data[2]), np.array(train_data[3])
        return train_image, train_label, train_bounding_box, train_filename, val_image, val_label, val_bounding_box, val_filename



def cross_validation_data_restore(cross_validation_number):
    """
    Statement:
    - 儲存訓練以及驗證資料。

    Parameters:
    - cross_validation_number: 第幾份data

    """

    train_image_path = f'D:/yolov7/datasets/cross{cross_validation_number}_condition/images/train/'
    train_label_path = f'D:/yolov7/datasets/cross{cross_validation_number}_condition/labels/train/'
    val_image_path = f'D:/yolov7/datasets/cross{cross_validation_number}_condition/images/val/'
    val_label_path = f'D:/yolov7/datasets/cross{cross_validation_number}_condition/labels/val/'

    for number, (image, label, bounding_box, filename) in enumerate(
            zip(train_image, train_label, train_bounding_box, train_filename)):
        image = cv2.resize(image, (512, 512), cv2.INTER_CUBIC)
        cv2.imwrite(train_image_path + f'{filename}_{number}.jpg', image * 255)
        with open(train_label_path + f'{filename}_{number}.txt', 'w') as train_label_file:
            for (categorical, points) in zip(label, bounding_box):
                train_label_file.write(str(categorical) + ' ')
                for p in points:
                    train_label_file.write(str(p) + " ")
                train_label_file.write('\n')

    for number, (image, label, bounding_box, filename) in enumerate(
            zip(val_image, val_label, val_bounding_box, val_filename)):
        image = cv2.resize(image, (512, 512), cv2.INTER_CUBIC)
        cv2.imwrite(val_image_path + f'{filename}_{number}.jpg', image * 255)
        with open(val_label_path + f'{filename}_{number}.txt', 'w') as val_label_file:
            for (categorical, points) in zip(label, bounding_box):
                val_label_file.write(str(categorical) + ' ')
                for p in points:
                    val_label_file.write(str(p) + " ")
                val_label_file.write('\n')

def cross_validation_data_write_path(cross_validation_number, data):
    write_path = f'D:/yolov7/datasets/cross{cross_validation_number}_condition/images/{data}/'
    read_path = f'D:/yolov7/datasets/cross{cross_validation_number}_condition/images/{data}/'
    target_path = f'D:/yolov7/datasets/cross{cross_validation_number}_condition/'
    with open(target_path + f'{data}.txt', 'w') as file:
        for filename in os.listdir(read_path):
            file.write(write_path + filename + '\n')


if __name__ == '__main__':
    data_method = YOLOv7_data_prepare(condition=False, cross_validation_number=5)
    data_method.cross_validation_data()