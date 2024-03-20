# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 15:04
# @Author  : Karry Ren

""" The preprocess code of raw I-SPY1 dataset (download from web).

Please `DOWNLOAD` I-SPY1 dataset following `README.md`,
    and move all files to `ISPY_DATASET_DOWNLOAD_PATH` to get the following directory structure:
    ISPY_DATASET_DOWNLOAD_PATH/
        ├── Dataset ISPY
        ├── outcome_new.xlsx
        └── Tumor_segmentation_new

Then you need to creat the following directory structure BY HAND:
    ISPY_DATASET_PROCESS_PATH/
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

The core operations of this preprocessing are:
    - split the dataset to train | valid | test.
    - slice the 3D images(masks) to 2d images(masks).
    - put the label to the image name.

Firstly you should set the `ISPY_DATASET_DOWNLOAD_PATH` and `ISPY_DATASET_PROCESS_PATH`
based on your situation, and set the hyper-param `KEEP_SLICE_NUM`.

Then you can run `python ispy_preprocess.py` to  preprocess the I-SPY1 Dataset.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# set the ISPY dataset path, the PATH is depended on your situation
# (we suggest you set the absolut path)
ISPY_DATASET_DOWNLOAD_PATH = "/Users/karry/KarryRen/Scientific-Projects/2023-UML/Code/Data/I-SPY1/ISPY_Download"
ISPY_DATASET_PROCESS_PATH = "/Users/karry/KarryRen/Scientific-Projects/2023-UML/Code/Data/I-SPY1/ISPY_Dataset"

# set the hyper-param `KEEP_SLICE_NUM`
# (we suggest you only keep 10 slices for each case)
KEEP_SLICE_NUM = 10

# ---- Step 1. Read the `SUBJECTID` and `PCR` from the `outcome_new.xlsx` ---- #
id_pcr0_tuple_list = []  # define the id and pcr(label=0) tuple empty list
id_pcr1_tuple_list = []  # define the id and pcr(label=1) tuple empty list
outcome_new_file_path = f"{ISPY_DATASET_DOWNLOAD_PATH}/outcome_new.xlsx"
outcome_new_df = pd.read_excel(outcome_new_file_path)
for row_idx, id_pcr_row in outcome_new_df.iterrows():
    id_pcr_tuple = (id_pcr_row["SUBJECTID"], id_pcr_row["PCR"])
    if id_pcr_row["PCR"] == 0:
        id_pcr0_tuple_list.append(id_pcr_tuple)
    elif id_pcr_row["PCR"] == 1:
        id_pcr1_tuple_list.append(id_pcr_tuple)
    else:
        raise TypeError(id_pcr_row["PCR"])
print(f"All PCR : 0-{len(id_pcr0_tuple_list)} cases, 1-{len(id_pcr1_tuple_list)} cases.")

# ---- Step 2. Split the train, valid and test ID ---- #
# Totally: 157 (114-pcr0, 43-pcr1) cases.
# Train: 127 (92-pcr0,  35-pcr1) cases;
# Valid: 15 (11-pcr0, 4-pcr1) cases;
# Test: 15 (11-pcr0, 4-pcr1) cases.
train_id_pcr_tuple_list = id_pcr0_tuple_list[:92] + id_pcr1_tuple_list[:35]
valid_id_pcr_tuple_list = id_pcr0_tuple_list[92:103] + id_pcr1_tuple_list[35:39]
test_id_pcr_tuple_list = id_pcr0_tuple_list[103:] + id_pcr1_tuple_list[39:]
print(f"Split: "
      f"Train-{len(train_id_pcr_tuple_list)} cases, "
      f"Valid-{len(valid_id_pcr_tuple_list)} cases, "
      f"Test-{len(test_id_pcr_tuple_list)} cases.")

# ---- Step 3. For-loop the id_pcr_tuple list and store .nii to .jpg images ---- #
for data_type in ["Train", "Valid", "Test"]:
    # get which id_pcr_tuple_list to read
    print(data_type)
    if data_type == "Train":
        id_pcr_tuple_list = train_id_pcr_tuple_list
    elif data_type == "Valid":
        id_pcr_tuple_list = valid_id_pcr_tuple_list
    elif data_type == "Test":
        id_pcr_tuple_list = test_id_pcr_tuple_list
    else:
        raise TypeError(data_type)
    # for loop the id_pcr_tuple_list
    for id_pcr_tuple in id_pcr_tuple_list:
        print(id_pcr_tuple)
        # get the subject id nad pcr
        sub_id = id_pcr_tuple[0]
        pcr = id_pcr_tuple[1]
        # construct the image and mask path
        image_path = f"{ISPY_DATASET_DOWNLOAD_PATH}/Dataset ISPY/ISPY_{sub_id}/DCEMRI_1.nii"
        mask_path = f"{ISPY_DATASET_DOWNLOAD_PATH}/Tumor_segmentation_new/ISPY_{sub_id}_tumor_mask.nii"
        # read the .nii image and mask
        image = nib.load(image_path)
        mask = nib.load(mask_path)
        # get the data and transform
        img_data = image.get_fdata()
        msk_data = mask.get_fdata()
        img_data = np.rot90(img_data, 1)
        img_data = np.flip(img_data, axis=0)
        msk_data = np.rot90(msk_data, 1)
        msk_data = np.flip(msk_data, axis=0)
        # assert shape equal
        assert img_data.shape == msk_data.shape, "img.shape != msk.shape, data ERROR !!"
        # make the case directories
        case_image_path = f"{ISPY_DATASET_PROCESS_PATH}/{data_type}/images/ispy_{sub_id}"
        case_mask_path = f"{ISPY_DATASET_PROCESS_PATH}/{data_type}/masks/ispy_{sub_id}"
        if not os.path.exists(case_image_path):
            os.makedirs(case_image_path)
        if not os.path.exists(case_mask_path):
            os.makedirs(case_mask_path)
        # compute max mask idx and get the start idx
        msk_sum = msk_data.sum(axis=(0, 1))
        msk_sum_idx = msk_sum.argmax()
        # for-loop the mask data slices
        slice_num = 0  # note the save slice num
        for slices in range(msk_sum_idx - KEEP_SLICE_NUM // 2, msk_data.shape[-1]):
            # just get the msk not all ground slices
            if not msk_data[:, :, slices].sum() == 0:
                # construct the process path
                process_image_path = f"{case_image_path}/ispy_{sub_id}_{pcr}_s{slices}_img.jpg"
                process_mask_path = f"{case_mask_path}/ispy_{sub_id}_{pcr}_s{slices}_msk.jpg"
                # save the image and mask
                plt.imsave(process_image_path, img_data[:, :, slices], cmap="gray")
                plt.imsave(process_mask_path, msk_data[:, :, slices], cmap="gray")
                slice_num += 1
            # just save KEEP_SLICE_NUM slices
            if slice_num == KEEP_SLICE_NUM:
                break
        # append data
        if slice_num < KEEP_SLICE_NUM:
            # just get the msk not all ground slices
            for ni in range(1, KEEP_SLICE_NUM):
                slices = msk_sum_idx - KEEP_SLICE_NUM // 2 - ni
                if not msk_data[:, :, slices].sum() == 0:
                    # construct the process path
                    process_image_path = f"{case_image_path}/ispy_{sub_id}_{pcr}_s{slices}_img.jpg"
                    process_mask_path = f"{case_mask_path}/ispy_{sub_id}_{pcr}_s{slices}_msk.jpg"
                    # save the image and mask
                    plt.imsave(process_image_path, img_data[:, :, slices], cmap="gray")
                    plt.imsave(process_mask_path, msk_data[:, :, slices], cmap="gray")
                    slice_num += 1
                # just save KEEP_SLICE_NUM slices
                if slice_num == KEEP_SLICE_NUM:
                    break
        assert slice_num == KEEP_SLICE_NUM, "Slice num WRONG !!"
