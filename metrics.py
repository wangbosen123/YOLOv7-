import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import re
import seaborn as sn
import random


class metrics():
    def __init__(self, val_image_path, val_gt_path, val_pred_path):
        self.val_image_path = val_image_path
        self.val_gt_path = val_gt_path
        self.val_pred_path = val_pred_path

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


    #計算裡個bounding box 的iou.
    def iou(self, box1, box2):
        """
            給定兩個box 計算兩者之間的overlap

            Parameters:
            - box1: 給定box1的點座標
            - box2: 給定box2的點座標

            Returns:
            - box1 and box2 的 IoU.
        """
        inter_left = max(box1[0], box2[0])
        inter_right = min(box1[0] + box1[2], box2[0] + box2[2])
        inter_top = max(box1[1], box2[1])
        inter_bottom = min(box1[1] + box1[3], box2[1] + box2[3])
        inter_area = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)
        union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
        iou = inter_area / union_area
        return iou


    #如何在圖像上根據bounding_box 畫出bounding box.
    def plot_boxx(self, image, bounding_box, gt=True):
        h, w = image.shape[0], image.shape[1]
        x1_, y1_, w1_, h1_ =  bounding_box[0],  bounding_box[1],  bounding_box[2],  bounding_box[3]
        x1_1 = w * x1_ - 0.5 * w * w1_
        x1_2 = w * x1_ + 0.5 * w * w1_
        y1_1 = h * y1_ - 0.5 * h * h1_
        y1_2 = h * y1_ + 0.5 * h * h1_
        tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)
        color = [random.randint(0, 255) for _ in range(3)]
        if gt:
            label_name = 'GT'
            color = [255, 0, 0]
        else:
            label_name = 'Pred'
            color = [0, 0, 255]
        cv2.putText(image, label_name, org=(int(x1_1), int(y1_1)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=tf, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (int(x1_1), int(y1_1)), (int(x1_2), int(y1_2)), color=color, thickness=3)
        return image, x1_1, y1_1, x1_2-x1_1, y1_2-y1_1


    def get_gt_output_list(self, filename):
        # Ground Truth Bounding Box.
        with open(self.val_gt_path + '/' + filename, "r") as f:
            content = f.read()
        lines = content.split('\n')
        gt_output_list = []
        for inputs in lines:
            for num, x in enumerate(re.split(' ', inputs)):
                if num > 0:
                    try:
                        gt_output_list.append(float(x))
                    except:
                        pass
        return gt_output_list


    def get_pred_output_list(self, filename):
        try:
            pred_output_list = []
            with open(self.val_pred_path + filename, "r") as f:
                content = f.read()
            lines = content.split('\n')
            for inputs in lines:
                pred_output = []
                for num, x in enumerate(re.split(' ', inputs)):
                    if num > 0:
                        try:
                            pred_output.append(float(x))
                        except:
                            pass
                if len(pred_output) > 0:
                    pred_output_list.append(pred_output)
            return pred_output_list
        except:
            return []


    def based_on_slices_space(self):
        val_image_path = os.listdir(self.val_image_path)
        val_image_path = sorted(val_image_path, key=self.natural_sort_key)

        val_pred_path = os.listdir(self.val_pred_path)
        val_pred_path = sorted(val_pred_path, key=self.natural_sort_key)

        TP, TN, FP, FN = 0, 0, 0, 0
        tumor_num, nontumor_num = 0, 0
        for filename in val_image_path:
            if '_tumor' in filename:
                tumor_num += 1
                if f'{filename[0:-4]}.txt' in val_pred_path:
                    TP += 1
                else:
                    FN += 1
            elif '_nontumor' in filename:
                nontumor_num += 1
                if f'{filename[0:-4]}.txt' in val_pred_path:
                    FP += 1
                else:
                    TN += 1
        print(tumor_num, nontumor_num)
        print('----------------')
        print(f'TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}')
        print(f'Accuracy is {(TP + TN) / (TP + TN + FP + FN)}, {TP + TN}, {TP + TN + FP + FN}')
        print(f'Recall is {TP / (TP + FN)}, {TP}, {TP + FN}')
        print(f'Precision is {TP / (TP + FP)}, {TP}, {TP + FP}')
        print(f'Specificity is {TN / (TN + FP)}, {TN}, {TN + FP}')
        print(f'NPV is {TN / (TN + FN)}, {TN}, {TN + FN}')
        return TP, TN, FP, FN


    def based_on_boundingbox_space(self, iou_threshold, iterate=False):
        val_image_path = os.listdir(self.val_image_path)
        val_image_path = sorted(val_image_path, key=self.natural_sort_key)

        val_pred_path = os.listdir(self.val_pred_path)
        val_pred_path = sorted(val_pred_path, key=self.natural_sort_key)

        TP, TN, FP, FN = 0, 0, 0, 0
        for filename in val_image_path:
            image = cv2.imread(self.val_image_path + filename[0: -4] + '.jpg', 0)
            # Compute TP, FN.
            if '_tumor' in filename:
                gt_output_list = self.get_gt_output_list(f'{filename[0:-4]}.txt')
                pred_output_list = self.get_pred_output_list(f'{filename[0:-4]}.txt')
                if pred_output_list == []:
                    FN += 1

                gt_image, gt_x1, gt_y1, gt_w, gt_h = self.plot_boxx(image, gt_output_list, gt=True)
                gt_box = [gt_x1, gt_y1, gt_w, gt_h]

                iou_record = []
                for num, pred in enumerate(pred_output_list):
                    pred_image, pred_x1, pred_y1, pred_w, pred_h = self.plot_boxx(image, pred, gt=False)
                    pred_box = [pred_x1, pred_y1, pred_w, pred_h]
                    iou_value = self.iou(gt_box, pred_box)
                    #                 print(filename, iou_value)
                    #                 plt.imshow(pred_image, cmap='gray')
                    #                 plt.show()

                    if len(pred_output_list) == 1:
                        if iou_value >= iou_threshold:
                            TP += 1
                        else:
                            if iterate:
                                FP += 1
                            FN += 1

                    elif len(pred_output_list) > 1:
                        iou_record.append(iou_value)
                        if (num + 1) == len(pred_output_list):
                            if max(iou_record) >= iou_threshold:
                                TP += 1
                                if iterate:
                                    FP += (len(iou_record) - 1)
                                FN += (len(iou_record) - 1)
                            else:
                                if iterate:
                                    FP += (len(iou_record))
                                FN += (len(iou_record))

                                # Compute TN, FP
            if '_nontumor' in f'{filename[0:-4]}.txt':
                TN += 1

            if (f'{filename[0:-4]}.txt' in val_pred_path) and '_nontumor' in f'{filename[0:-4]}.txt':
                with open(self.val_pred_path + f'{filename[0:-4]}.txt', "r") as f:
                    content = f.read()
                lines = content.split('\n')
                FP += (len(lines) - 1)
                TN -= 1

        print('----------------')
        print(f'TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}')
        print(f'Accuracy is {(TP + TN) / (TP + TN + FP + FN)}, {TP + TN}, {TP + TN + FP + FN}')
        print(f'Recall is {TP / (TP + FN)}, {TP}, {TP + FN}')
        print(f'Precision is {TP / (TP + FP)}, {TP}, {TP + FP}')
        print(f'Specificity is {TN / (TN + FP)}, {TN}, {TN + FP}')
        print(f'NPV is {TN / (TN + FN)}, {TN}, {TN + FN}')
        return TP, FN, TN, FP



    def based_on_tumor_space(self, iou_threshold, conse, lower_limit, upper_limit):
        val_image_path = os.listdir(self.val_image_path)
        val_image_path = sorted(val_image_path, key=self.natural_sort_key)

        val_pred_path = os.listdir(self.val_pred_path)
        val_pred_path = sorted(val_pred_path, key=self.natural_sort_key)


        val_path = []
        for filename in val_image_path:
            if 'nontumor' in filename:
                if lower_limit < int(filename[9:-13]) < upper_limit:
                    val_path.append(filename)
            if '_tumor' in filename:
                if lower_limit < int(filename[9:-10]) < upper_limit:
                    val_path.append(filename)

        val_image_path = val_path

        def has_consecutive_n(arr, n):
            if len(arr) < n:
                return False

            for i in range(len(arr) - n + 1):
                if all(arr[i + j] == arr[i] + j for j in range(n)):
                    return True
            return False

        FP_case_number = []
        TP_case_number = []
        FN_case_number = []

        order = []
        case_number = []
        case_number.append(int(val_image_path[0][5:8]))

        non_order = []
        non_case_number = []
        non_case_number.append(int(val_image_path[0][5:8]))

        total_case_number = []
        total_case_number.append(int(val_image_path[0][5:8]))

        for filename in val_image_path:
            image = cv2.imread(self.val_image_path + filename[0: -4] + '.jpg', 0)

            if int(filename[5: 8]) not in total_case_number:
                total_case_number.append(int(filename[5:8]))

            if int(filename[5:8]) not in case_number:
                if has_consecutive_n(order, n=conse):
                    TP_case_number.append(case_number[-1])
                else:
                    FN_case_number.append(case_number[-1])

                order = []
                case_number.append(int(filename[5:8]))

            if '_tumor' in filename:
                gt_output_list = self.get_gt_output_list(f'{filename[0:-4]}.txt')
                pred_output_list = self.get_pred_output_list(f'{filename[0:-4]}.txt')

                gt_image, gt_x1, gt_y1, gt_w, gt_h = self.plot_boxx(image, gt_output_list, gt=True)
                gt_box = [gt_x1, gt_y1, gt_w, gt_h]

                iou_record = []
                for num, pred in enumerate(pred_output_list):
                    pred_image, pred_x1, pred_y1, pred_w, pred_h =self. plot_boxx(image, pred, gt=False)
                    pred_box = [pred_x1, pred_y1, pred_w, pred_h]
                    iou_value = self.iou(gt_box, pred_box)

                    if len(pred_output_list) == 1:
                        if iou_value >= iou_threshold:
                            order.append(int(filename[9: -10]))

                    elif len(pred_output_list) > 1:
                        iou_record.append(iou_value)
                        if (num + 1) == len(pred_output_list):
                            if max(iou_record) >= iou_threshold:
                                order.append(int(filename[9:-10]))

            if filename is val_image_path[-1]:
                if has_consecutive_n(order, n=conse):
                    TP_case_number.append(case_number[-1])
                else:
                    FN_case_number.append(case_number[-1])

            # Compute TN, FP
            if (f'{filename[0:-4]}.txt' in val_pred_path) and '_nontumor' in f'{filename[0:-4]}.txt':
                if int(filename[5:8]) not in non_case_number:
                    if has_consecutive_n(non_order, n=conse):
                        FP_case_number.append(non_case_number[-1])
                    non_order = []
                    non_case_number.append(int(filename[5:8]))

                non_order.append(int(filename[9:-13]))
        print('----------------')
        TP = len(TP_case_number)
        FN = len(total_case_number) - len(TP_case_number)
        TN = len(total_case_number) - len(FP_case_number)
        FP = len(FP_case_number)
        print(f'TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}')
        print(f'Accuracy is {(TP + TN) / (TP + TN + FP + FN)}, {TP + TN}, {TP + TN + FP + FN}')
        print(f'Recall is {TP / (TP + FN)}, {TP}, {TP + FN}')
        print(f'Precision is {TP / (TP + FP)}, {TP}, {TP + FP}')
        print(f'Specificity is {TN / (TN + FP)}, {TN}, {TN + FP}')
        print(f'NPV is {TN / (TN + FN)}, {TN}, {TN + FN}')
        return TP, FN, TN, FP, TP_case_number, FN_case_number, FP_case_number


if __name__ == '__main__':
    detection_metrics = metrics(val_image_path='D:/yolov7/inference/cross5/images/val/', val_gt_path='D:/yolov7/inference/cross5/labels/val', val_pred_path='D:/yolov7/runs/detect/cross5/labels/')
    # detection_metrics.based_on_slices_space()
    detection_metrics.based_on_boundingbox_space(iou_threshold=0.001, iterate=True)
    # for i in range(1, 7):
    #     TP3, FN3, TN3, FP3, TP_case, FN_case, FP_case = detection_metrics.based_on_tumor_space(iou_threshold=0.001, conse=i, lower_limit=13, upper_limit=65)
