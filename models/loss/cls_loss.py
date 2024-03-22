# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 18:47
# @Author  : Karry Ren

""" The loss function for classification. Ref. TMC. """

import torch
import torch.nn.functional as F


def KL(alpha, c):
    """
    Args:
        alpha: the Dirichlet of cls
        c: num of cls classes

    Returns: KL loss of cls

    """

    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def cls_evidence_loss(p, alpha, c, current_epoch, annealing_epoch):
    """
    Args:
        p: label of cls result (bs, 1)
        alpha: the Dirichlet of cls (bs, c)
        c: classes of cls
        current_epoch: train_epoch (changing while train)
        annealing_epoch: set in config

    Returns:the overall loss of classification (ace_loss + lamda * kl_loss)

    """

    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)

    # ---- Loss 1. L_ace ---- #
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # ---- Loss 2. L_KL ---- #
    annealing_coef = min(1, current_epoch / annealing_epoch)  # gradually increase from 0 to 1
    alp = E * (1 - label) + 1
    L_kL = annealing_coef * KL(alp, c)

    return torch.mean(L_ace + L_kL)


def cls_ce_loss(pred_cls, target):
    """
    Args:
        pred_cls: the output from cls model (just a score no need softmax and log) (bs, cls)
        target: label of cls result (bs, 1)

    Returns: the cross entropy loss

    """

    # ---- no need Softmax ! ---- #
    return F.cross_entropy(pred_cls, target)
