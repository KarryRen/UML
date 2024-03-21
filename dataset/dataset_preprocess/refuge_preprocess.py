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
        ├── Valid
        └── Test

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
import nibabel as nib

# set the ISPY dataset path, the PATH is depended on your situation
# (we suggest you set the absolut path)
ISPY_DATASET_DOWNLOAD_PATH = "/Users/karry/KarryRen/Scientific-Projects/2023-UML/Code/Data/I-SPY1/ISPY_Download"
ISPY_DATASET_PROCESS_PATH = "/Users/karry/KarryRen/Scientific-Projects/2023-UML/Code/Data/I-SPY1/ISPY_Dataset"

# set the hyper-param `KEEP_SLICE_NUM`
# (we suggest you only keep 10 slices for each case)
KEEP_SLICE_NUM = 10

# begin pre-process
print(f"===================== I-SPY1 Begin Pre-Process =====================")

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
    print(f"============ {data_type} =============")
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

print(f"===================== I-SPY1 Pre-Process Over ! =====================")
