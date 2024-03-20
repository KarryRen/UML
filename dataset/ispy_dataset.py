# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 16:42
# @Author  : Karry Ren

""""""

import torch.utils.data
import os
import numpy as np
import cv2


def file_extend(PATH_root, file_name_list):
    extend_file_name_list = []
    for i in range(len(file_name_list)):
        extend_file_name_list.append(PATH_root + "\\" + file_name_list[i])
    return extend_file_name_list


class JointBreastDataset(torch.utils.data.Dataset):
    """ get breast data """

    def __init__(self, path_breast_data, path_breast_fold, fold_num, data_type='train'):
        self.data_type = data_type

        # ---- image | gt | name filename list for train | valid | test ---- #
        self.case_list = []
        self.case_list_0 = []
        self.case_list_1 = []
        self.img_list = []
        self.mask_list = []

        path_breast_fold_num = path_breast_fold + '\\fold' + fold_num
        # - step 1 get case filename list from fold txt
        with open(path_breast_fold_num + "\\" + data_type + ".txt", encoding='utf-8') as file:
            cases_content = file.readlines()
        for case_content in cases_content:
            case_name = case_content.split("\t")[0]
            case_label = case_content.split("\t")[1].split("\n")[0]
            # - step 2 get case folder name
            case_file = 'ISPY_' + case_name + '_' + case_label
            self.case_list.append(case_file)
            if case_label == '0':
                self.case_list_0.append(case_file)
            else:
                self.case_list_1.append(case_file)
            # - step 3 get case images and masks folder name
            case_img_folder_name = path_breast_data + '\\' + case_file + '\\' + case_file + '_image'
            case_msk_folder_name = path_breast_data + '\\' + case_file + '\\' + case_file + '_mask'
            case_img_name_list = os.listdir(case_img_folder_name)
            case_msk_name_list = os.listdir(case_msk_folder_name)
            # - step 4 get full path name
            case_img_full_name_list = file_extend(case_img_folder_name, case_img_name_list)
            case_msk_full_name_list = file_extend(case_msk_folder_name, case_msk_name_list)
            assert len(case_img_full_name_list) == len(case_msk_full_name_list)
            # - step 5 extend path of images and masks
            self.img_list.extend(case_img_full_name_list)
            self.mask_list.extend(case_msk_full_name_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = plt.imread(self.img_list[idx])
        mask = cv2.imread(self.mask_list[idx], 0)
        label = int(self.img_list[idx].split("_")[-3])
        image_name = self.img_list[idx].split("\\")[-1]
        mask_name = self.mask_list[idx].split("\\")[-1]
        image = image.transpose((2, 0, 1)).astype(np.float32)  # change data type to FloatTensor

        # - must do to avoid pixel != 0 or != 1
        mask = (mask != 0).astype(np.int64)  # change data type to LongTensor

        return image / 255, mask, label, image_name

    def get_img_info(self, idx):
        image = cv2.imread(self.img_list[idx])
        return {"height": image.shape[0], "width": image.shape[1]}
