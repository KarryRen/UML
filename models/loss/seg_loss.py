import torch
import logging
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
from torch.nn.modules.loss import _Loss


def KL(alpha, c):
    """
    :param alpha: the Dirichlet of seg
    :param c: num of seg classes
    :return: KL loss of seg
    """
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    return kl


def SDiceLoss(preds, targets, weight_map=None):
    """
    :param preds: pred segmentation (bs, c, h, w)
    :param targets: segmentation mask (bs, h, w, c)
    :param weight_map: optional
    :return: DICE loss
    """
    C = preds.size(1)  # classes of seg
    num_class = C
    if preds.ndim == 5:
        preds = preds.permute(0, 2, 3, 4, 1)
    else:
        preds = preds.permute(0, 2, 3, 1)  # (bs, h, w, c)

    pred = preds.contiguous().view(-1, num_class)  # (bs*h*w, c)
    ground = targets.view(-1, num_class)  # (bs*h*w, c)

    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        intersect = torch.sum(ground * pred, 0)
        ref_vol = torch.sum(ground, 0)
        seg_vol = torch.sum(pred, 0)

    dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1e-5)
    dice_mean_score = torch.mean(-torch.log(dice_score))

    return dice_mean_score


def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label (pixel-wise one hot and change dim)
        input_tensor: tensor with shape [bs, 1, h, w]
        output_tensor: tensor with shape [bs, h, w, num_seg_class]
    """
    tensor_list = []
    if input_tensor.ndim == 5:
        input_tensor = input_tensor.permute(0, 2, 3, 4, 1)
    else:
        input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):  # 0, 1, 2 ...
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()

    return output_tensor


def seg_evidence_loss(p, alpha, c, current_epoch, annealing_epoch, evidence):
    """
    :param p: mask of seg result (bs, 1, h, w)
    :param alpha: the Dirichlet of seg (bs, c, h, w)
    :param c: classes of seg
    :param current_epoch: current_epoch: train_epoch (changing while train)
    :param annealing_epoch: set in config
    :param evidence: the evidence of seg (bs, c, h, w)
    :return: the overall loss of t_seg
    """
    # ---- LOSS 1. L_dice ---- #
    evidence = torch.softmax(evidence, dim=1)  # compute the probability of seg result (bs, c, h, w)
    soft_p = get_soft_label(p, c)  # one hot mask with dim trans for SDiceloss (bs, h, w, c)
    L_dice = SDiceLoss(evidence, soft_p)

    # ---- LOSS 2. L_ice ---- #
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # (bs, c, h*w)
    alpha = alpha.transpose(1, 2)  # (bs, h*w, c)
    alpha = alpha.contiguous().view(-1, alpha.size(2))  # (bs*h*w, c)
    S = torch.sum(alpha, dim=1, keepdim=True)  # (bs*h*w, 1)
    E = alpha - 1  # evidence
    one_hot_label = F.one_hot(p, num_classes=c)  # (bs, 1, h, w, c)
    seg_label = one_hot_label.view(-1, c)  # (bs*h*w, c)
    L_ice = torch.sum(seg_label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    L_ice = torch.mean(L_ice)

    # ---- LOSS 3. L_KL ---- #
    annealing_coef = min(1, current_epoch / annealing_epoch)
    alp = E * (1 - seg_label) + 1
    L_KL = annealing_coef * KL(alp, c)
    L_KL = torch.mean(L_KL)

    return L_ice + L_KL + L_dice


def seg_dice_loss(pred_seg, target, c):
    """
    :param pred_seg: the output from cls model (just a score no need softmax) (bs, 2)
    :param target: gt
    :param c: classes of seg
    :return: the dice loss
    """
    preds = torch.softmax(pred_seg, dim=1)  # compute the probability of seg result (bs, c, h, w)
    soft_target = get_soft_label(target, c)  # one hot mask with dim trans for SDiceloss (bs, h, w, c)
    L_dice = SDiceLoss(preds, soft_target)

    return L_dice


if __name__ == '__main__':
    image = torch.rand((2, 2, 256, 256)).to(torch.float32)
    mask = torch.randint(0, 2, (2, 1, 256, 256)).to(torch.int64)
    print(mask[0][0][128][0])

    label = torch.Tensor([0, 1]).to(torch.int64)
    print(label.type())
    seg_evidence = torch.rand((2, 2, 256, 256)).to(torch.float32)

    seg_alpha = seg_evidence + 1
    # a = seg_evidence_loss(p=mask, alpha=seg_alpha, c=2, current_epoch=1, annealing_epoch=1, evidence=seg_evidence)
    # print(a)
    test_dice = seg_dice_loss(pred_seg=image, target=mask, c=2)
    print(test_dice)  # NOTE:CORRECT
