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

    # 分割train 以及 val
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
        #分割整體資料集為5等份，指定第cross_validation_number為validation data 其餘為training data.
        for num, filename in enumerate(os.listdir(path)):
            if (self.cross_validation_number - 1) * 60 <= num < (self.cross_validation_number - 1) * 60 + 60:
                val_name.append(int(filename[5: 8]))
            else:
                train_name.append(int(filename[5: 8]))

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
                            image = self.apply_window_level_width(image, window_center=20, window_width=180)

                            if self.condition:
                                # 新增condition 目的為增加前後文資訊量。
                                concate_image1 = raw_image[:, :, num_image]
                                concate_image2 = raw_image[:, :, num_image - 2]
                                concate_image1 = self.apply_window_level_width(concate_image1, window_center=20, window_width=180)
                                concate_image2 = self.apply_window_level_width(concate_image2, window_center=20, window_width=180)
                                image = image.reshape(image.shape[0], image.shape[1], 1)
                                concate_image1 = concate_image1.reshape(concate_image1.shape[0], concate_image2.shape[1], 1)
                                concate_image2 = concate_image2.reshape(concate_image2.shape[0], concate_image2.shape[1], 1)
                                image = np.concatenate((concate_image1, image, concate_image2), axis=-1)

                            # 如果需要回傳資料，將下面註解拔掉。
                            # train_image.append(image)
                            # train_filename.append(filename[0:8])
                            # train_bounding_box.append([])
                            # train_label.append([])
                            # train_bounding_box[-1].append(pos1)
                            # train_label[-1].append(0)

                            cv2.imwrite(train_image_path + f'{filename[0:8]}_{train_number}.jpg', image * 255)
                            with open(train_label_path + f'{filename[0:8]}_{train_number}.txt', 'w') as train_label_file:
                                for (categorical, points) in zip([0], [pos1]):
                                    train_label_file.write(str(categorical) + ' ')
                                    for p in points:
                                        train_label_file.write(str(p) + " ")
                                    train_label_file.write('\n')
                            train_number += 1

                        if int(filename[5: 8]) in val_name:
                            image = raw_image[:, :, num_image - 1]
                            image = self.apply_window_level_width(image, window_center=20, window_width=180)

                            if self.condition:
                                # 新增condition 目的為增加前後文資訊量。
                                concate_image1 = raw_image[:, :, num_image]
                                concate_image2 = raw_image[:, :, num_image - 2]
                                concate_image1 = self.apply_window_level_width(concate_image1, window_center=20, window_width=180)
                                concate_image2 = self.apply_window_level_width(concate_image2, window_center=20, window_width=180)
                                image = image.reshape(image.shape[0], image.shape[1], 1)
                                concate_image1 = concate_image1.reshape(concate_image1.shape[0], concate_image2.shape[1], 1)
                                concate_image2 = concate_image2.reshape(concate_image2.shape[0], concate_image2.shape[1], 1)
                                image = np.concatenate((concate_image1, image, concate_image2), axis=-1)

                            # 如果需要回傳資料，將下面註解拔掉。
                            # val_image.append(image)
                            # val_filename.append(filename[0:8])
                            # val_bounding_box.append([])
                            # val_label.append([])
                            # val_bounding_box[-1].append(pos1)
                            # val_label[-1].append(0)

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

        # 如果需要回傳資料，將下面註解拔掉。
        #     train_image, train_label, train_bounding_box, train_filename = np.array(train_image), np.array(train_label), np.array(train_bounding_box), np.array(train_filename)
        #     val_image, val_label, val_bounding_box, val_filename = np.array(val_image), np.array(val_label), np.array(val_bounding_box), np.array(val_filename)
        #     print(train_image.shape, train_label.shape, train_bounding_box.shape, train_filename.shape)
        #     print(val_image.shape, val_label.shape, val_bounding_box.shape, val_filename.shape)
        #     print(train_filename)
        #     train_data = list(zip(train_image, train_label, train_bounding_box, train_filename))
        #     np.random.shuffle(train_data)
        #     train_data = list(zip(*train_data))
        #     train_image, train_label, train_bounding_box, train_filename = np.array(train_data[0]), np.array(train_data[1]), np.array(train_data[2]), np.array(train_data[3])
        # return train_image, train_label, train_bounding_box, train_filename, val_image, val_label, val_bounding_box, val_filename

    # 將train 以及 val 整理成一個txt檔案，YOLOv7會需要
    def cross_validation_data_write_path(self):
        for type in ['train', 'val']:
            write_path = f'D:/yolov7/datasets/cross{self.cross_validation_number}/images/{type}/'
            read_path = f'D:/yolov7/datasets/cross{self.cross_validation_number}/images/{type}/'
            target_path = f'D:/yolov7/datasets/cross{self.cross_validation_number}/'
            with open(target_path + f'{type}.txt', 'w') as file:
                for filename in os.listdir(read_path):
                    file.write(write_path + filename + '\n')

    def validation_data_all(self, cross_validation_number):
        path = 'C:/Users/CMUHCH_AIAMIC/Desktop/overall data/ROI data/'
        val_name = []
        for num, filename in enumerate(os.listdir(path)):
            if (cross_validation_number - 1) * 60 <= num < (cross_validation_number - 1) * 60 + 60:
                val_name.append(int(filename[5: 8]))

        val_name = np.array(val_name)
        # 直接讀取一個volume 用於validation.
        val_image_path = f'D:/yolov7/inference/cross{cross_validation_number}/images/val/'
        val_label_path = f'D:/yolov7/inference/cross{cross_validation_number}/labels/val/'
        path = f'C:/Users/CMUHCH_AIAMIC/Desktop/overall data/ROI data/'


        for num_filename, filename in enumerate(os.listdir(path)):
            if int(filename[5: 8]) in val_name:
                print('--------')
                print(filename)
                data = loadmat(path + filename)  # 读取mat文件
                raw_image = data['dwi']
                #  print(raw_image.shape)

                yoloposition = data['yoloposition']
                volume_count = 1
                tumor_index = []
                val_image, val_label, val_bounding_box, val_filename, val_nontumor_filename = [], [], [], [], []
                for i in range(yoloposition.shape[0]):
                    num_image = yoloposition[i][1][0][0]
                    if num_image == -1:
                        continue

                    try:
                        pos1 = yoloposition[i][0][0][0][0]
                        if pos1[0] != 0:
                            tumor_index.append(num_image)
                            val_filename.append(filename[0:8])
                            val_bounding_box.append([])
                            val_bounding_box[-1].append(pos1)
                    except:
                        pass

                bounding_box = sorted(zip(tumor_index, val_bounding_box), key=lambda x: x[0])
                bounding_box = list(zip(*bounding_box))
                bounding_box = list(bounding_box[1])
                tumor_index.sort()
                volume_index = [i for i in range(1, raw_image.shape[-1] + 1)]

                count = 0
                for index in volume_index:
                    if (index != volume_index[0]) or (index != volume_index[-1]):
                        if index == raw_image.shape[-1]:
                            continue
                        #                         print(index-1, index, index-2)
                        image = raw_image[:, :, index - 1]
                        image = self.apply_window_level_width(image, window_center=20, window_width=180)

                        # 新增condition 目的為增加前後文資訊量。
                        if self.condition:
                            concate_image1 = raw_image[:, :, index]
                            concate_image2 = raw_image[:, :, index - 2]
                            concate_image1 = self.apply_window_level_width(concate_image1, window_center=20, window_width=180)
                            concate_image2 = self.apply_window_level_width(concate_image2, window_center=20, window_width=180)
                            image = image.reshape(image.shape[0], image.shape[1], 1)
                            concate_image1 = concate_image1.reshape(concate_image1.shape[0], concate_image2.shape[1], 1)
                            concate_image2 = concate_image2.reshape(concate_image2.shape[0], concate_image2.shape[1], 1)
                            image = np.concatenate((concate_image1, image, concate_image2), axis=-1)

                        #                 image = raw_image[:, :, index-1]
                        #                 image = apply_window_level_width(image,  window_center=20, window_width=180)
                        label = [0]
                        if index in tumor_index:
                            cv2.imwrite(val_image_path + f'{filename[0:8]}_{volume_count}_tumor.jpg', image * 255)
                            with open(val_label_path + f'{filename[0:8]}_{volume_count}_tumor.txt', 'w') as val_label_file:
                                for (categorical, points) in zip(label, bounding_box[count]):
                                    val_label_file.write(str(categorical) + ' ')
                                    for p in points:
                                        val_label_file.write(str(p) + " ")
                                    val_label_file.write('\n')
                            count += 1

                        else:
                            cv2.imwrite(val_image_path + f'{filename[0:8]}_{volume_count}_nontumor.jpg', image * 255)
                    volume_count += 1




if __name__ == '__main__':
    data_method = YOLOv7_data_prepare(condition=False, cross_validation_number=5)
    # data_method.cross_validation_data()
    # data_method.cross_validation_data_write_path()
    data_method.validation_data_all(cross_validation_number=5)
