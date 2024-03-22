# -*- coding: utf-8 -*-
# @Time    : 2023/3/22 16:08
# @Author  : Karry Ren

""" The util functions. """

import random
from typing import Optional
import torch.optim
import matplotlib.pyplot as plt


def print_log(log_info: str, log_file=None) -> None:
    """ Print the log information to console and log_file.

    :param log_info: the information to log
    :param log_file: the file to log

    """

    # ---- Step 1. Print the log_info to the log_file ---- #
    if log_file is not None:
        # use print function
        print(log_info, file=log_file)
        # flush the file
        if random.randint(0, 20) < 3:
            log_file.flush()

    # ---- Step 2. Print the log_info to the console ---- #
    print(log_info)


def adjust_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, epoch: int, whole_steps: int,
              base_steps: int, power: float, change_epoch: Optional[int] = 30) -> None:
    """ Adjust the learning rate.

    :param optimizer: the optimizer
    :param base_lr: the initial learning rate
    :param step: the current step
    :param epoch: the current epoch
    :param whole_steps: the whole steps
    :param base_steps: the base train steps
    :param power: lr down power
    :param change_epoch: the change epoch

    Returns:
        the new lr for one step

    """

    # ---- Change the lr based on the steps ---- #
    if change_epoch is not None:  # have the change epoch
        if epoch >= change_epoch:  # after the change epoch, start changing (be smaller)
            new_lr = base_lr * ((1 - float(step - base_steps) / (whole_steps - base_steps)) ** power)
        else:  # before the change epoch, keep lr
            new_lr = base_lr
    else:  # not have the change epoch, keep changing
        new_lr = base_lr * ((1 - float(step) / whole_steps) ** power)

    # ---- Bound lr it to 1e-5 ---- #
    if new_lr <= 1e-5:
        new_lr = 1e-5

    # ---- Set the lr to optimizer ---- #
    optimizer.param_groups[0]["lr"] = new_lr


def draw_curves(data_list, label_list, color_list, linestyle_list=None, filename='training_curve.png'):
    plt.figure()
    for i in range(len(data_list)):
        data = data_list[i]
        label = label_list[i]
        color = color_list[i]
        if linestyle_list == None:
            line_style = '-'
        else:
            line_style = linestyle_list[i]
        plt.plot(data, label=label, color=color, linestyle=line_style)
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()
    plt.close()
    plt.show()
    plt.close('all')
