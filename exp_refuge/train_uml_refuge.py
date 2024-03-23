# -*- coding: utf-8 -*-
# @Time    : 2023/3/22 15:41
# @Author  : Karry Ren

""" The UML_Net training code of Refuge Dataset. """

import os
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch
import time
import numpy as np
import warnings

import config_refuge as config
from datasets.refuge_dataset import JointRefugeDataset
from models.uml_net import UML_Net
from utils import print_log, adjust_lr, draw_curves
from models.loss.cls_loss import cls_evidence_loss
from models.loss.seg_loss import seg_dice_loss, seg_ce_loss, seg_evidence_loss

# ---- Ignore Warnings ---- #
warnings.filterwarnings("ignore")

# ---- Setting CUDA Environment ---- #
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID  # set the visible devices
cudnn.enabled, cudnn.benchmark = True, True  # set the cuda config

# ---- Build up the file directory and build the log file ---- #
if not os.path.isdir(config.PATH_SAVE):
    os.mkdir(config.PATH_SAVE)
if not os.path.isdir(config.PATH_CHECKPOINTS):
    os.mkdir(config.PATH_CHECKPOINTS)
if not os.path.isdir(config.PATH_LOGFILE):
    os.mkdir(config.PATH_LOGFILE)
if not os.path.isdir(config.PATH_MODEL):
    os.mkdir(config.PATH_MODEL)
log_file_path = config.PATH_LOGFILE + "training_output.log"
logfile = open(log_file_path, "a")

# ---- Print the information for console and logfile ---- #
print_log(os.getcwd(), logfile)
print_log(f"************ Device: {os.environ['CUDA_VISIBLE_DEVICES']} ======= ", logfile)
print_log(f"************ Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ************", logfile)
print_log(f"************ Settings ************", logfile)
print_log(f"************ LR: {config.LEARNING_RATE}; BS: {config.BATCH_SIZE} ************", logfile)

# ---- Make Train and Valid dataset ---- #
train_loader = data.DataLoader(JointRefugeDataset(root_path=config.REFUGE_ROOT_PATH, data_type="Train"),
                               batch_size=config.BATCH_SIZE, shuffle=True)  # train dataloader
valid_loader = data.DataLoader(JointRefugeDataset(root_path=config.REFUGE_ROOT_PATH, data_type="Valid"),
                               batch_size=1, shuffle=False)  # valid dataloader
train_t_loader = data.DataLoader(JointRefugeDataset(root_path=config.REFUGE_ROOT_PATH, data_type="Train"),
                                 batch_size=1, shuffle=False)  # 1 batch size train loader for testing
print_log(f"************ Train image num: {len(train_t_loader)} ************", logfile)
print_log(f"************ Valid image num: {len(valid_loader)} ************", logfile)

# ---- Construct the NetWork, Optimizer and Empty Loss-List --- #
model = UML_Net(pretrained_res2net_path=config.PRETRAINED_RES2NET_PATH,
                seg_class=config.SEG_CLASS, cls_class=config.CLS_CLASS)
model = torch.nn.DataParallel(model).cuda()  # make the model parallel to cuda
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
sum_loss_list_all_epoch, cls_loss_list_all_epoch, seg_loss_list_all_epoch, mut_loss_list_all_epoch = [], [], [], []
epoch_valid_metric = []

# ---- Train && Valid Epoch by Epoch ---- #
for epoch in range(config.EPOCH):
    # one epoch loss list
    sum_loss_list_one_epoch = []  # loss of sum (cls + seg + mut)
    cls_loss_list_one_epoch = []  # loss of cls
    seg_loss_list_one_epoch = []  # loss of seg
    mut_loss_list_one_epoch = []  # loss of mut

    # start the training iter by iter
    for i_iter, batch_data in enumerate(train_loader):
        # compute the training step
        step = (config.TRAIN_NUM / config.BATCH_SIZE) * epoch + i_iter
        # get the data
        images = batch_data["image"].to(dtype=torch.float32).cuda()  # images, shape=(bs, 3, h, w)
        cls_labels = batch_data["cls_label"].to(dtype=torch.int64).cuda()  # cls_labels, shape=(bs, 1)
        seg_gts = batch_data["seg_gt"].to(dtype=torch.int64).cuda()  # seg_gts, shape=(bs, 1, h, w)
        # train
        optimizer.zero_grad()  # zero the optimizer grad
        adjust_lr(optimizer=optimizer, base_lr=config.LEARNING_RATE, step=step, epoch=epoch,
                  whole_steps=config.WHOLE_STEPS, base_steps=config.BASE_STEPS,
                  power=config.POWER, change_epoch=config.CHANGE_EPOCH)  # adjust the lr of optimizer
        # forward
        model.train()
        cls_alpha, cls_uncertainty, mut_evidence_list, mut_alpha_list, mut_uncertainty_list, final_seg = model(images)
        # compute loss
        # - cls loss
        cls_loss = cls_evidence_loss(cls_labels, cls_alpha, config.CLS_CLASS, epoch, config.CLS_ANNEALING_EPOCH)
        # - seg loss
        seg_loss = seg_dice_loss(seg_gts, final_seg, config.SEG_CLASS) + seg_ce_loss(seg_gts, final_seg)
        # - mut loss
        mut_loss = seg_evidence_loss(seg_gts, mut_alpha_list[0], config.SEG_CLASS, epoch, config.SEG_ANNEALING_EPOCH,
                                     mut_evidence_list[0])
        for i in range(1, len(mut_alpha_list)):
            mut_loss += seg_evidence_loss(seg_gts, mut_alpha_list[i], config.SEG_CLASS, epoch,
                                          config.CLS_ANNEALING_EPOCH, mut_evidence_list[i])
        # backward
        sum_loss = (config.CLS_LOSS_WEIGHT * cls_loss + config.SEG_LOSS_WEIGHT * seg_loss
                    + config.MUT_LOSS_WEIGHT * mut_loss)
        sum_loss.backward()
        optimizer.step()

        # list the loss for one epoch
        sum_loss_list_one_epoch.append(sum_loss.cpu().data.numpy())
        cls_loss_list_one_epoch.append(cls_loss.cpu().data.numpy())
        seg_loss_list_one_epoch.append(seg_loss.cpu().data.numpy())
        mut_loss_list_one_epoch.append(mut_loss.cpu().data.numpy())

    # training log
    line = "Train-Epoch [%d/%d] [All]: Loss = %.6f, Cls_loss = %.6f, Seg_loss = %.6f, Mut_loss = %.6f, LR = %0.9f" % (
        epoch, config.EPOCH, np.nanmean(sum_loss_list_one_epoch), np.nanmean(cls_loss_list_one_epoch),
        np.nanmean(seg_loss_list_one_epoch), np.nanmean(mut_loss_list_one_epoch), optimizer.param_groups[0]["lr"])
    print_log(line, logfile)

    # note the loss for one epoch
    sum_loss_list_all_epoch.append(np.nanmean(sum_loss_list_one_epoch))
    cls_loss_list_all_epoch.append(np.nanmean(cls_loss_list_one_epoch))
    seg_loss_list_all_epoch.append(np.nanmean(seg_loss_list_one_epoch))
    mut_loss_list_all_epoch.append(np.nanmean(mut_loss_list_one_epoch))

    # save model and do validation
    # if epoch >= config.VALID_EPOCH:
    #     # 1. save model
    #     torch.save(model.state_dict(), f"{config.PATH_MODEL}/{epoch}_uml_refuge")
    #     # 2. eval the valid loader
    #     valid_result = eval_UML_REFUGE(valid_data_loader=valid_loader, model=model)
    #     # -- cls
    #     [AUC, ACC, F1] = valid_result['cls']
    #     line = "Valid-EVAL-EPOCH [%d/%d] [Cls]: AUC = %f, ACC = %f, F1 = %f" % (
    #         epoch, config.REFUGE_EPOCH, AUC, ACC, F1)
    #     print_f(line, f=logfile)
    #     # -- seg
    #     [DICE1, ASSD1, DICE2, ASSD2] = valid_result['seg']
    #     line = "Valid-EVAL-EPOCH [%d/%d] [Seg]: DICE1 = %f, ASSD1 = %f, DICE2 = %f, ASSD2 = %f " % (
    #         epoch, config.REFUGE_EPOCH, DICE1, ASSD1, DICE2, ASSD2)
    #     print_f(line, f=logfile)
    #     # -- cls seg two task index
    #     CLS_SEG_valid.append(ACC + AUC + F1 + DICE1 + DICE2)
    #     print("---------------------------------------------------------------------------")
    #     print()

# ---- Plot Loss curve ---- #
loss_curve_file_name = f"{config.PATH_LOGFILE}/loss_curves_uml.png"
data_list = [sum_loss_list_all_epoch, cls_loss_list_all_epoch, seg_loss_list_all_epoch, mut_loss_list_all_epoch]
label_list = ['sum_loss', 'cls_loss', 'seg_loss', 'mut_loss']
draw_curves(data_list=data_list, label_list=label_list, color_list=config.COLOR[0:4], filename=loss_curve_file_name)
print("---------------------------------------------------------------------------")
