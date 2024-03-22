# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 18:49
# @Author  : Karry Ren

""" The loss function for segmentation. Ref. TBraTS. """

import torch
import torch.nn.functional as F


def KL(alpha, c):
    """
    Compute the KL Loss of Segmentation
    Args:
        alpha: the pixel-wise alpha of seg (bs, c, h, w)
        c: the classes of seg

    Returns:
        the KL loss  of pixel-wise alpha
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
    compute the Dice Loss of pixel-wise Seg Result
    Args:
        preds: pred pixel-wise seg result (bs, c, h, w)
        targets: seg GT after one-hot (bs, h, w, c)
        weight_map: the weighted average map (h, w) if just average, make it None

    Returns:
        average Dice Loss of c classes seg
        the average means that:
            if the class of segmentation is c, then the preds.shape = (bs, c, h, w) and the targets.shape = (bs, h, w, c)
            the dice will be computed one by one class, just like
                - 1. compute dice score of bg (c = 0)
                - 2. compute dice score of c1 (c = 1)
                ...
                - 3. compute dice score of cn (c = n - 1)
            then sum and do averaging
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
    convert GT to one-hot pixel-wise label
    Args:
        input_tensor: GT (bs, 1, h, w)
        num_class: the number of seg classes

    Returns:
        one-hot GT (bs, h, w, num_class)
    """

    tensor_list = []

    if input_tensor.ndim == 5:
        input_tensor = input_tensor.permute(0, 2, 3, 4, 1)
    else:
        input_tensor = input_tensor.permute(0, 2, 3, 1)  # to (bs, h, w, c)

    for i in range(num_class):  # 0, 1, 2 ...
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))  # do one-hot
        tensor_list.append(temp_prob)

    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()

    return output_tensor


def seg_evidence_loss(p, alpha, c, current_epoch, annealing_epoch, evidence):
    """
    get the evidence loss of Trusted Seg
    Args:
        p: mask of seg result (bs, 1, h, w)
        alpha: the Dirichlet of seg (bs, c, h, w)
        c: classes of seg
        current_epoch: current_epoch: train_epoch (changing while train)
        annealing_epoch: set in config
        evidence:  the evidence of seg (bs, c, h, w)

    Returns:
        the overall loss of trusted seg
    """

    # ---- Loss 1. L_dice ---- #
    evidence = torch.softmax(evidence, dim=1)  # compute the probability of seg result (bs, c, h, w)
    soft_p = get_soft_label(p, c)  # one hot mask with dim trans for SDiceloss (bs, h, w, c)
    L_dice = SDiceLoss(evidence, soft_p)

    # ---- Loss 2. L_ice ---- #
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # (bs, c, h*w)
    alpha = alpha.transpose(1, 2)  # (bs, h*w, c)
    alpha = alpha.contiguous().view(-1, alpha.size(2))  # (bs*h*w, c)
    S = torch.sum(alpha, dim=1, keepdim=True)  # (bs*h*w, 1)
    E = alpha - 1  # evidence
    one_hot_label = F.one_hot(p, num_classes=c)  # (bs, 1, h, w, c)
    seg_label = one_hot_label.view(-1, c)  # (bs*h*w, c)
    L_ice = torch.sum(seg_label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    L_ice = torch.mean(L_ice)

    # ---- Loss 3. L_KL ---- #
    annealing_coef = min(1, current_epoch / annealing_epoch)
    alp = E * (1 - seg_label) + 1
    L_KL = annealing_coef * KL(alp, c)
    L_KL = torch.mean(L_KL)

    return L_dice + L_ice + L_KL


def seg_dice_loss(target, pred_seg, c):
    """
    get The Dice Loss of Seg Result
    Args:
        pred_seg: the output from seg model (just a score no need softmax) (bs, c, h, w)
        target: gt
        c: classes of seg

    Returns:
        the dice loss
    """

    preds = torch.softmax(pred_seg, dim=1)  # compute the probability of seg result (bs, c, h, w)
    soft_target = get_soft_label(target, c)  # one-hot mask with dim trans for SDiceloss (bs, h, w, c)
    L_dice = SDiceLoss(preds, soft_target)  # get the dice loss

    return L_dice


def seg_ce_loss(target, pred_seg):
    """
    get The Cross Entrophy Loss of Seg Result
    Args:
        pred_seg: the output from seg model (just a score no need softmax) (bs, c)
        target: gt

    Returns:
        the dice loss
    """

    # ---- no need softmax ! ---- #
    target = target.squeeze(1)
    L_ce = F.cross_entropy(pred_seg, target)

    return L_ce


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

    test_dice = seg_ce_loss(pred_seg=image, target=mask)
    print(test_dice)  # NOTE:CORRECT
