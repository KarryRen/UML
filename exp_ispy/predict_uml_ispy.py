# -*- coding: utf-8 -*-
# @Time    : 2023-03-23 22:52
# @Author  : Kai Ren
# @Comment : used for test data

import torch
import os
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

import exp_ispy.config_ispy as config
from models.uml_net import UML_Net
from datasets.ispy_dataset import JointIspyDataset

if not os.path.isdir(config.SEG_RESULT_PATH_OF_TEST):  # test path
    os.mkdir(config.SEG_RESULT_PATH_OF_TEST)

# ---- Setting cuda environment ---- #
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_TEST


def predict_refuge():
    # ---- set data loader or just one image ---- #
    test_loader = data.DataLoader(JointIspyDataset(root_path=config.ISPY_ROOT_PATH, data_type="Train"),
                                  batch_size=1, shuffle=False)
    print("test image num:", len(test_loader))

    # ---- load trained model ---- #
    model = UML_Net(pretrained_res2net_path=config.PRETRAINED_RES2NET_PATH,
                    seg_class=config.SEG_CLASS, cls_class=config.CLS_CLASS).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("E:/renkai/Joint_RK/Code/UML/exp_ispy/save/checkpoints/Model_results/9_uml_ispy"))

    # ---- predict and get index ---- #
    result = predict_ispy_refuge(test_data_loader=test_loader, model=model,
                                 save_img_path=config.SEG_RESULT_PATH_OF_TEST)


def predict_ispy_refuge(test_data_loader, model, save_img_path):
    # cls index list
    pred_label_list = []
    true_label_list = []
    true_label_one_hot_list = []

    # seg index list
    seg_dice_list = []
    seg_assd_list = []

    for index, batch_data in enumerate(test_data_loader):
        image = batch_data["image"].to(dtype=torch.float32)  # images, shape=(bs, 3, h, w)
        raw_test_images = image.numpy()
        image_noises_Gaussian = np.random.normal(0, 0.2, raw_test_images.shape)
        noisy_test_images = raw_test_images + image_noises_Gaussian
        noisy_test_images = np.clip(noisy_test_images, 0.0, 1.0).astype(np.float32)
        image = torch.from_numpy(noisy_test_images).cuda()

        cls_label = batch_data["cls_label"].to(dtype=torch.int64).cuda()  # cls_labels, shape=(bs)
        seg_gt = batch_data["seg_gt"].to(dtype=torch.int64).cuda()  # seg_gts, shape=(bs, 1, h, w)
        item_name = batch_data["item_name"][0]

        # ---- compute output ---- #
        model.eval()
        cls_alpha, cls_uncertainty, mut_evidence_list, mut_alpha_list, mut_uncertainty_list, final_seg = model(image)

        # ---- classification pred result ---- #
        pred_label = torch.softmax(cls_alpha[0], dim=0).cpu().data.numpy()
        num_classes = pred_label.shape[-1]  # the number of classes
        true_label = cls_label[0].cpu().data.numpy()
        true_label_one_hot = F.one_hot(cls_label[0], num_classes=num_classes).cpu().data.numpy()
        pred_label_list.append(pred_label)  # after softmax -> (2)
        true_label_list.append(true_label)  # true label -> (1)
        true_label_one_hot_list.append(true_label_one_hot)

        # ---- segmentation pred result ---- #
        pred_seg_out_raw = torch.softmax(final_seg, dim=1).cpu().data.numpy()
        pred_seg_out_arg = np.argmax(pred_seg_out_raw[0], axis=0)

        plt.imsave(item_name, image[0].cpu().data.numpy().transpose(1, 2, 0), dpi=1000)
        plt.imsave("seg.jpg", pred_seg_out_arg, cmap="gray", dpi=1000)

        if index == 2:
            print(pred_label)
            break


if __name__ == '__main__':
    predict_refuge()
