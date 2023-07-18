import torch.utils.data
import os
import numpy as np
import cv2
import config as config


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
        image = cv2.imread(self.img_list[idx])
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


if __name__ == '__main__':
    database_train = JointBreastDataset(path_breast_data=config.BREAST_DATA_FILE,
                                        path_breast_fold=config.BREAST_FOLD_FILE,
                                        fold_num=config.BREAST_FOLD_NUM,
                                        data_type='train')

    database_valid = JointBreastDataset(path_breast_data=config.BREAST_DATA_FILE,
                                        path_breast_fold=config.BREAST_FOLD_FILE,
                                        fold_num=config.BREAST_FOLD_NUM,
                                        data_type='valid')

    database_test = JointBreastDataset(path_breast_data=config.BREAST_DATA_FILE,
                                       path_breast_fold=config.BREAST_FOLD_FILE,
                                       fold_num=config.BREAST_FOLD_NUM,
                                       data_type='test')

    test_no_cross = set(database_train.case_list) & set(database_valid.case_list) & set(database_test.case_list)
    print("==== test no cross ====")
    print("cross file list = ", list(test_no_cross))

    print("==== test case num ==== ")
    print("case of train = ", len(database_train.case_list), " case of 0 = ", len(database_train.case_list_0),
          " case of 1 = ", len(database_train.case_list_1))
    print("case of valid = ", len(database_valid.case_list), " case of 0 = ", len(database_valid.case_list_0),
          " case of 1 = ", len(database_valid.case_list_1))
    print("case of test = ", len(database_test.case_list), " case of 0 = ", len(database_test.case_list_0),
          " case of 1 = ", len(database_test.case_list_1))

    print(" ==== count image num ==== ")
    train_loader = torch.utils.data.DataLoader(database_train, batch_size=1, num_workers=0,
                                               shuffle=False, pin_memory=False, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(database_valid, batch_size=1, num_workers=0,
                                               shuffle=False, pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(database_test, batch_size=1, num_workers=0,
                                              shuffle=False, pin_memory=False, drop_last=True)
    print("all image num of train = ", len(train_loader))
    sum0, sum1 = 0, 0
    for img, seg, label, image_name in train_loader:
        if label == 0:
            sum0 += 1
        elif label == 1:
            sum1 += 1
    print("case label 0  = ", sum0, "case label 1 = ", sum1)
    print()
    print("all image num of valid = ", len(valid_loader))
    sum0, sum1 = 0, 0
    for img, seg, label, image_name in valid_loader:
        print(image_name)
        if label == 0:
            sum0 += 1
        elif label == 1:
            sum1 += 1
    print("case label 0  = ", sum0, "case label 1 = ", sum1)
    print()
    print("all image num of test = ", len(test_loader))
    sum0, sum1 = 0, 0
    for img, seg, label, image_name in test_loader:
        if label == 0:
            sum0 += 1
        elif label == 1:
            sum1 += 1
    print("case label 0  = ", sum0, "case label 1 = ", sum1)
