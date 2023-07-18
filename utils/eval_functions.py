import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
import config
from models.loss import *
from utils.visualization import *
import utils.binary as binary


# ---- REFUGE ---- #
def eval_UML_REFUGE(valid_data_loader, model, epoch, save_seg_path=None, verbose=False, whole_epoch=0):
    # cls index list
    pred_label_list = []
    true_label_list = []
    pred_score_list = []

    # seg index list
    seg_dice1_list = []
    seg_assd1_list = []
    seg_dice2_list = []
    seg_assd2_list = []

    # steps
    whole_steps = len(valid_data_loader)
    steps = 0
    for index, test_batch in enumerate(valid_data_loader):
        pred_seg_arg_list = []

        # ---- Prepare Input Data ---- #
        test_images, test_masks, test_label, test_name = test_batch
        test_images = test_images.cuda()
        test_label = test_label.cuda()
        test_mask = test_masks[0].data.numpy()

        # ---- compute output ---- #
        model.eval()
        with torch.no_grad():
            cls_alpha, cls_uncertainty, \
            mutual_evidence, mutual_alpha, mutual_uncertainty, \
            seg_list, seg_list_view, reliable_mask_list = model(test_images)

        # ---- classification pred result ---- #
        pred_label = torch.softmax(cls_alpha[0], dim=0).cpu().data.numpy()
        true_label = test_label[0].cpu().data.numpy()
        pred_label_list.append(pred_label)  # after softmax -> (2)
        true_label_list.append(true_label)  # true label -> (1)
        pred_score_list.append(pred_label[1])  # score(label=1)

        # ---- segmentation pred result ---- #
        for pred_seg_view in seg_list_view:
            pred_seg_raw = torch.softmax(pred_seg_view, dim=1).cpu().data.numpy()
            pred_seg_arg = np.argmax(pred_seg_raw[0], axis=0)
            pred_seg_arg_list.append(pred_seg_arg)

        mutual_uncertainty_map = mutual_uncertainty[0, 0, :, :].cpu().data.numpy()  # seg uncertainty
        pred_mut_alpha_raw = torch.softmax(mutual_alpha, dim=1).cpu().data.numpy()
        pred_mut_alpha_arg = np.argmax(pred_mut_alpha_raw[0], axis=0)

        pred_seg_out_raw = torch.softmax(seg_list[0], dim=1).cpu().data.numpy()
        pred_seg_out_arg = np.argmax(pred_seg_out_raw[0], axis=0)
        seg_dice1_list.append(binary.dc((pred_seg_out_arg == 1), (test_mask == 1)))  # DICE
        seg_assd1_list.append(binary.assd((pred_seg_out_arg == 1), (test_mask == 1)))  # ASSD
        seg_dice2_list.append(binary.dc((pred_seg_out_arg == 2), (test_mask == 2)))  # DICE
        seg_assd2_list.append(binary.assd((pred_seg_out_arg == 2), (test_mask == 2)))  # ASSD

        steps += 1

        if verbose == config.VERBOSE and epoch == whole_epoch - 1 and steps > whole_steps * 0.9:
            show_img_gt_segList_mut_un(img=test_images[0].cpu().data.numpy().transpose(1, 2, 0),
                                       gt=test_mask,
                                       seg_list=pred_seg_arg_list,
                                       mut_alpha=pred_mut_alpha_arg,
                                       mut_un=mutual_uncertainty_map,
                                       save_path=save_seg_path,
                                       name=test_name[0])

    # ---- compute index of classification && segmentation ---- #
    pred_label_array = np.array(pred_label_list)  # list lens = bs, element shape (2) the softmax(cls_output)
    true_label_array = np.array(true_label_list)  # list lens = bs, element shape (2)
    pred_score_array = np.array(pred_score_list)
    preds = np.argmax(np.array(pred_label_array), axis=-1)
    ACC = metrics.accuracy_score(true_label_array, preds)  # AUC
    F1 = metrics.f1_score(true_label_array, preds, average='weighted')  # F1

    DICE1 = np.nanmean(seg_dice1_list)  # DICE for seg
    ASSD1 = np.nanmean(seg_assd1_list)  # ASSD for seg
    DICE2 = np.nanmean(seg_dice2_list)  # DICE for seg
    ASSD2 = np.nanmean(seg_assd2_list)  # ASSD for seg

    # ---- summarise index of cls and seg ---- #
    result = {}
    result['cls'] = [ACC, F1]
    result['seg'] = [DICE1, ASSD1, DICE2, ASSD2]

    return result


# ---- BREAST ---- #
def eval_UML_BREAST(valid_data_loader, model, epoch, save_seg_path=None, verbose=False, whole_epoch=0):
    """
        eval the JCSUnet
    """
    # cls index list
    pred_label_list = []
    true_label_list = []
    pred_score_list = []

    # seg index list
    seg_dice_list = []
    seg_assd_list = []

    # steps
    whole_steps = len(valid_data_loader)
    steps = 0
    for index, test_batch in enumerate(valid_data_loader):
        pred_seg_arg_list = []
        pred_rmp_arg_list = []

        # ---- Prepare Input Data ---- #
        test_images, test_masks, test_label, test_name = test_batch
        test_images = test_images.cuda()
        test_label = test_label.cuda()
        test_mask = test_masks[0].data.numpy()

        # ---- compute output ---- #
        model.eval()
        with torch.no_grad():
            cls_alpha, cls_uncertainty, \
            mutual_evidence, mutual_alpha, mutual_uncertainty, \
            seg_list, seg_list_view, reliable_mask_list = model(test_images)

        # ---- classification pred result ---- #
        pred_label = torch.softmax(cls_alpha[0], dim=0).cpu().data.numpy()
        true_label = test_label[0].cpu().data.numpy()
        pred_label_list.append(pred_label)  # after softmax -> (2)
        true_label_list.append(true_label)  # true label -> (1)
        pred_score_list.append(pred_label[1])  # score(label=1)

        # ---- segmentation pred result ---- #
        for pred_seg_view in seg_list_view:
            pred_seg_raw = torch.softmax(pred_seg_view, dim=1).cpu().data.numpy()
            pred_seg_arg = np.argmax(pred_seg_raw[0], axis=0)
            pred_seg_arg_list.append(pred_seg_arg)

        mutual_uncertainty_map = mutual_uncertainty[0, 0, :, :].cpu().data.numpy()  # seg uncertainty
        pred_mut_alpha_raw = torch.softmax(mutual_alpha, dim=1).cpu().data.numpy()
        pred_mut_alpha_arg = np.argmax(pred_mut_alpha_raw[0], axis=0)

        pred_seg_out_raw = torch.softmax(seg_list[0], dim=1).cpu().data.numpy()
        pred_seg_out_arg = np.argmax(pred_seg_out_raw[0], axis=0)
        seg_dice_list.append(binary.dc(pred_seg_out_arg, test_mask))  # DICE
        seg_assd_list.append(binary.assd(pred_seg_out_arg, test_mask))  # ASSD

        steps += 1

        if verbose == config.VERBOSE and epoch == whole_epoch - 1 and steps > whole_steps * 0.9:
            show_img_gt_segList_mut_un(img=test_images[0].cpu().data.numpy().transpose(1, 2, 0),
                                       gt=test_mask,
                                       seg_list=pred_seg_arg_list,
                                       mut_alpha=pred_mut_alpha_arg,
                                       mut_un=mutual_uncertainty_map,
                                       save_path=save_seg_path,
                                       name=test_name[0])

    # ---- compute index of classification && segmentation ---- #
    pred_label_array = np.array(pred_label_list)  # list lens = bs, element shape (2) the softmax(cls_output)
    true_label_array = np.array(true_label_list)  # list lens = bs, element shape (2)
    pred_score_array = np.array(pred_score_list)
    preds = np.argmax(np.array(pred_label_array), axis=-1)
    ACC = metrics.accuracy_score(true_label_array, preds)  # AUC
    F1 = metrics.f1_score(true_label_array, preds, average='weighted')  # F1

    DICE = np.nanmean(seg_dice_list)  # DICE for seg
    ASSD = np.nanmean(seg_assd_list)  # ASSD for seg

    # ---- summarise index of cls and seg ---- #
    result = {}
    result['cls'] = [ACC, F1]
    result['seg'] = [DICE, ASSD]

    return result
