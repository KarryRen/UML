# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 15:05
# @Author  : Karry Ren

""" The preprocess code of raw Refuge dataset (download from web).

Because we want to make the code clear and beautiful, so we need you to
    do some directory creation and unzip !!!

Please `DOWNLOAD` Refuge dataset following `README.md`. You need unzip the `.zip` files
    and move all files to `REFUGE_DATASET_DOWNLOAD_PATH` to get the following directory structure:
    REFUGE_DATASET_DOWNLOAD_PATH/
        ├── Train
            ├── Train400 (unzip from `Training400.zip`) Please
            └── Train400-GT (unzip from `Annotation-Training400.zip`)
        ├── Valid
            ├── Valid400 (unzip from `REFUGE-Validation400.zip`)
            └── Valid400-GT (unzip from `REFUGE-Validation400-GT.zip`)
        └── Test
            ├── Test400 (unzip from `Test400.zip`)
            └── Test400-GT  (unzip from `REFUGE-Test-GT.zip`)
    please format the directory structure to be same !!!

Then you need to create the following directory structure `BY HAND`:
    REFUGE_DATASET_PROCESS_PATH/
        ├── Train
            ├── images
            └── masks
        ├── Valid
            ├── images
            └── masks
        └── Test
            ├── images
            └── masks

The core operations of this preprocessing are:
    - put the label to the image and mask name.
    - resize the image and task to target size.

Firstly you should set the `REFUGE_DATASET_DOWNLOAD_PATH` and `REFUGE_DATASET_PROCESS_PATH`
based on your situation, and set the hyper-param `IMAGE_SIZE`.

Then you can run `python refuge_preprocess.py` to  preprocess the Refuge Dataset and
    you will get the following directory structure:
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

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# set the Refuge dataset path, the PATH is depended on your situation
# (we suggest you set the absolut path)
REFUGE_DATASET_DOWNLOAD_PATH = "/Users/karry/KarryRen/Scientific-Projects/2023-UML/Code/Data/Refuge/Refuge_Download"
REFUGE_DATASET_PROCESS_PATH = "/Users/karry/KarryRen/Scientific-Projects/2023-UML/Code/Data/Refuge/Refuge_Dataset"

# set the hyper-param `IMAGE_SIZE`
# (we suggest you set the 256x256 image_size)
IMAGE_SIZE = 256

# begin pre-process
print(f"===================== Refuge Begin Pre-Process =====================")

# ---- For-loop the data_type and resize each image&mask ---- #
for data_type in ["Train", "Valid", "Test"]:
    print(f"============ {data_type} =============")

    # construct the image and mask root path
    image_root_path = f"{REFUGE_DATASET_DOWNLOAD_PATH}/{data_type}/{data_type}400"
    mask_root_path = f"{REFUGE_DATASET_DOWNLOAD_PATH}/{data_type}/{data_type}400-GT/Disc_Cup_Masks"
    # construct the target path
    image_process_root_path = f"{REFUGE_DATASET_PROCESS_PATH}/{data_type}/images"
    mask_process_root_path = f"{REFUGE_DATASET_PROCESS_PATH}/{data_type}/masks"
    # read the `Glaucoma_label_and_Fovea_location.xlsx` (please rename the `.xlsx` file in train and valid)
    label_df = pd.read_excel(f"{REFUGE_DATASET_DOWNLOAD_PATH}/{data_type}/{data_type}400-GT/"
                             f"Glaucoma_label_and_Fovea_location.xlsx")
    # for loop to process data
    for row_idx, label_df_row in label_df.iterrows():
        # get the id and label
        refuge_id = label_df_row["ImgName"].split(".")[0]
        refuge_label = label_df_row["Glaucoma Label"]
        print(f"Refuge ID: {refuge_id}, Refuge Label: {refuge_label}.")
        # construct the image&mask file
        image_file_path = f"{image_root_path}/{refuge_id}.jpg"
        mask_file_path = f"{mask_root_path}/{refuge_id}.bmp"
        # set the target file path
        image_process_path = f"{image_process_root_path}/{refuge_id}_{refuge_label}_img.jpg"
        mask_process_path = f"{mask_process_root_path}/{refuge_id}_{refuge_label}_msk.bmp"
        # read the image&mask data and resize the data
        image = cv2.imread(image_file_path)
        mask = cv2.imread(mask_file_path)
        # resize the image&mask
        resized_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        resized_mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        # stored the pre-processed new image
        cv2.imwrite(image_process_path, resized_image)
        cv2.imwrite(mask_process_path, resized_mask)

print(f"===================== Refuge Pre-Process Over ! =====================")
