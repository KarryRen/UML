import torch
import torch.nn.functional as F


# ---- CLS LOSS ----#
# ---- refer to TMC ----#
def KL(alpha, c):
    """
    :param alpha: the Dirichlet of cls
    :param c: num of cls classes
    :return: KL loss of cls
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
    :param p: label of cls result (bs, 1)
    :param alpha: the Dirichlet of cls (bs, 2)
    :param c: classes of cls
    :param current_epoch: train_epoch (changing while train)
    :param annealing_epoch: set in config
    :return: the overall loss of classification (ace_loss + lamda * kl_loss)
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    # the adjust cross-entropy loss
    ace_loss = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # the KL loss
    annealing_coef = min(1, current_epoch / annealing_epoch)  # gradually increase from 0 to 1
    alp = E * (1 - label) + 1
    kl_loss = annealing_coef * KL(alp, c)

    return torch.mean(ace_loss + kl_loss)


def ce_loss(pred_cls, target):
    """
    :param pred_cls: the output from cls model (just a score no need softmax and log) (bs, 2)
    :param target: label of cls result (bs, 1)
    :return: the ce loss
    """
    return F.cross_entropy(pred_cls, target)
