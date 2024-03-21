# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 15:09
# @Author  : Karry Ren

""" The torch.Dataset of Refuge dataset.

After the preprocessing raw Refuge dataset (download from web) by
    run `python refuge_preprocess.py` you will get the following Refuge dataset directory:
    REFUGE_DATASET_PROCESS_PATH/
        ├── Train
            ├── images
                ├── refugeID1_glaucoma(0 or 1)_img.jpg
                ├── refugeID2_glaucoma_img.jpg
                ├── ...
                └── refugeIDn_glaucoma_img.jpg
            └── masks
                ├── refugeID1_glaucoma(0 or 1)_msk.jpg
                ├── refugeID2_glaucoma_msk.jpg
                ├── ...
                └── refugeIDn_glaucoma_msk.jpg
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


class JointRefugeDataset(torch.utils.data.Dataset):
    """ The Joint Refuge Dataset. """

    def __init__(self, root_path: str, data_type: str = "Train"):
        """ The init function of JointRefugeDataset. Will load all image and mask file path.

        :param root_path: the root path of Refuge Dataset
        :param data_type: the data type, you have only 3 choices:
            - "Train" for train data
            - "Valid" for valid data
            - "Test" for test data

        """

        # ---- Step 1. Define the images and masks file path list ---- #
        self.images_file_path_list = []
        self.masks_file_path_list = []

        # ---- Step 2. Get all images and masks path ---- #
        # construct the images and masks directory path
        images_directory_path = f"{root_path}/{data_type}/images"
        masks_directory_path = f"{root_path}/{data_type}/masks"
        # get the case id in directory
        for image_file_path in sorted(os.listdir(images_directory_path)):
            self.images_file_path_list.append(f"{images_directory_path}/{image_file_path}")
        for mask_file_path in sorted(os.listdir(masks_directory_path)):
            self.masks_file_path_list.append(f"{masks_directory_path}/{mask_file_path}")

        # ---- Step 3. Check Data Len ---- #
        assert len(self.images_file_path_list) == len(self.masks_file_path_list), "Image Mask num not total equal !!!"

    def __len__(self):
        """ Get the length of dataset. """

        return len(self.images_file_path_list)

    def __getitem__(self, idx: int):
        """ Get the item.

        :param idx: the item idx

        return: a dict with the format:
            {
                "image": the image array, shape=(3, 128, 128),
                "cls_label": the label for classification, shape=(1,),
                "seg_gt": the ground truth for segmentation, shape=(1, 128, 128),
                    only have 0, 1 and 2, 0-gd, 1-cup and 2-disc
                "item_name": a str,
            }
        """

        # ---- Check image and mask right ---- #
        image_name = self.images_file_path_list[idx].split("/")[-1]
        image_id = image_name.split("_")[1]
        mask_name = self.masks_file_path_list[idx].split("/")[-1]
        mask_id = mask_name.split("_")[1]
        assert image_id == mask_id, "Image Mask not Right !!!"

        # ---- Read the image, label and mask ---- #
        # - image
        image = plt.imread(self.images_file_path_list[idx]).copy()  # shape=(h, w, 3)
        image = (image / 255).transpose(2, 0, 1)  # scale to [0, 1], and transpose to (3, h, w)
        assert (0.0 <= image).all() and (image <= 1.0).all(), "image value ERROR !!!"
        # - label for classification task
        cls_label = np.array([int(image_name.split("_")[1])])  # shape=(1,)
        # - gt for segmentation task, and make it to 0-gd, 1-cup, 2-disc
        seg_gt = plt.imread(self.masks_file_path_list[idx])[:, :, 0].copy()  # shape=(h, w)
        seg_gt[seg_gt < 10] = 2
        seg_gt[(seg_gt >= 10) & (seg_gt <= 245)] = 1
        seg_gt[seg_gt > 245] = 0
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


if __name__ == "__main__":  # a demo using JointRefugeDataset
    REFUGE_DATASET_PATH = "/Users/karry/KarryRen/Scientific-Projects/2023-UML/Code/Data/Refuge/Refuge_Dataset"

    refuge_dataset = JointRefugeDataset(root_path=REFUGE_DATASET_PATH, data_type="Test")

    # show the image
    print(refuge_dataset[1]["item_name"])
    plt.subplot(1, 2, 1)
    plt.imshow(refuge_dataset[1]["image"].transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(refuge_dataset[1]["seg_gt"])
    plt.show()

    for i in range(len(refuge_dataset)):
        print(refuge_dataset[i]["image"].max(), refuge_dataset[i]["image"].min())
        print(refuge_dataset[i]["seg_gt"].sum())
        print(refuge_dataset[i]["cls_label"])
