# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 16:42
# @Author  : Karry Ren

""" The torch.Dataset of I-SPY1 dataset.

After the preprocessing raw I-SPY1 dataset (download from web) by
    run `python ispy_preprocess.py` you will get the following I-SPY1 dataset directory:
    ISPY_DATASET_PATH/
        ├── Train
            ├── images
                ├── ispy_xxxx1 (KEEP_SLICE_NUM images)
                    ├── ispy_xxxx1_pcr_s1_img.jpg
                    ├── ispy_xxxx1_pcr_s2_img.jpg
                    ├── ...
                    └── ispy_xxxx1_pcr_sn_img.jpg
                ├── ispy_xxxx2
                ├── ...
                └── ispy_xxxxn
            └── masks
                ├── ispy_xxxx1 (KEEP_SLICE_NUM masks)
                    ├── ispy_xxxx1_pcr_s1_msk.jpg
                    ├── ispy_xxxx1_pcr_s2_msk.jpg
                    ├── ...
                    └── ispy_xxxx1_pcr_sn_msk.jpg
                ├── ispy_xxxx2
                ├── ...
                └── ispy_xxxxn
        ├── Valid
            ├── images
            └── masks
        └── Test
            ├── images
            └── masks

In this dataset:
    - during `__init__()`, we will LOAD all images and masks file path.
    - during `__getitem__()`, we will READ 1 image and mask and label to memory.

"""

import torch.utils.data
import os
import numpy as np
import matplotlib.pyplot as plt


class JointIspyDataset(torch.utils.data.Dataset):
    """ The Joint I-SPY1 Dataset. """

    def __init__(self, root_path: str, data_type: str = "Train"):
        """ The init function of JointIspyDataset. Will load all image and mask file path.

        :param root_path: the root path of I-SPY1 Dataset
        :param data_type: the data type, you have only 3 choices:
            - "Train" for train data
            - "Valid" for valid data
            - "Test" for test data

        """

        # ---- Step 1. Define the images and masks file path list ---- #
        self.images_file_path_list = []
        self.masks_file_path_list = []

        # ---- Step 2. Get all cases ---- #
        # construct the images and masks directory path
        images_directory_path = f"{root_path}/{data_type}/images"
        masks_directory_path = f"{root_path}/{data_type}/masks"
        # get the case id in directory
        images_case_id_list = sorted(os.listdir(images_directory_path))
        masks_case_id_list = sorted(os.listdir(masks_directory_path))
        assert images_case_id_list == masks_case_id_list, "images cases are not == masks cases !!!"
        case_id_list = images_case_id_list  # set the images_cases_list to cases_list

        # ---- Step 3. Read all images and masks file path ---- #
        for case_id in case_id_list:
            # get the cased images and masks directory
            images_case_id_directory_path = f"{images_directory_path}/{case_id}"
            masks_case_id_directory_path = f"{masks_directory_path}/{case_id}"
            # get all images and masks file path
            images_case_id_path_list = sorted(os.listdir(images_case_id_directory_path))
            masks_case_id_path_list = sorted(os.listdir(masks_case_id_directory_path))
            assert len(images_case_id_path_list) == len(masks_case_id_path_list), "Image Mask num not equal !!!"
            # append all path to list
            for images_case_id_path in images_case_id_path_list:
                self.images_file_path_list.append(f"{images_case_id_directory_path}/{images_case_id_path}")
            for masks_case_id_path in masks_case_id_path_list:
                self.masks_file_path_list.append(f"{masks_case_id_directory_path}/{masks_case_id_path}")

        # ---- Step 4. Check Data Len ---- #
        assert len(self.images_file_path_list) == len(self.masks_file_path_list), "Image Mask num not total equal !!!"

    def __len__(self):
        """ Get the length of dataset. """

        return len(self.images_file_path_list)

    def __getitem__(self, idx: int):
        """ Get the item.

        :param idx: the item idx

        return: a dict with the format:
            {
                "image": the image array, shape=(3, 128, 128)
                "cls_label": the label for classification, shape=(1,)
                "seg_gt": the ground truth for segmentation, shape=(1, 128, 128)
                    only have 0 and 1, 0-gd and 1-tumor
                "item_name": a str
            }
        """

        # ---- Check image and mask right ---- #
        image_name = self.images_file_path_list[idx].split("/")[-1]
        image_case_id = image_name.split("_")[1]
        image_slice_num = image_name.split("_")[3]
        mask_name = self.masks_file_path_list[idx].split("/")[-1]
        mask_case_id = mask_name.split("_")[1]
        mask_slice_num = mask_name.split("_")[3]
        assert image_case_id == mask_case_id and image_slice_num == mask_slice_num, "Image Mask not Right !!!"

        # ---- Read the image, label and mask ---- #
        # - image
        image = plt.imread(self.images_file_path_list[idx])  # shape=(h, w, 3)
        image = (image / 255).transpose(2, 0, 1)  # scale to [0, 1], and transpose to (3, h, w)
        assert (0.0 <= image).all() and (image <= 1.0).all(), "image value ERROR !!!"
        # - label for classification task
        cls_label = np.array([int(image_name.split("_")[2])])  # shape=(1,)
        # - gt for segmentation task, and make it 0-gd, 1-tumor
        seg_gt = plt.imread(self.masks_file_path_list[idx])[:, :, 0]  # shape=(h, w)
        seg_gt = (seg_gt >= 50).astype(np.int8)  # avoid not 0 or 1
        # - the item name, just be the image name
        item_name = image_name

        # ---- Construct the item ---- #
        item = {
            "image": image,
            "cls_label": cls_label,
            "seg_gt": seg_gt,
            "item_name": item_name
        }

        return item


if __name__ == "__main__":  # a demo using JointIspyDataset
    ISPY_DATASET_PATH = "/Users/karry/KarryRen/Scientific-Projects/2023-UML/Code/Data/I-SPY1/ISPY_Dataset"

    ispy_dataset = JointIspyDataset(root_path=ISPY_DATASET_PATH, data_type="Test")

    # show the image
    print(ispy_dataset[1]["item_name"])
    plt.subplot(1, 2, 1)
    plt.imshow(ispy_dataset[1]["image"].transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(ispy_dataset[1]["seg_gt"])
    plt.show()

    # for i in range(len(ispy_dataset)):
    #     print(ispy_dataset[i]["image"].max(), ispy_dataset[i]["image"].min())
    #     print(ispy_dataset[i]["seg_gt"].sum())
