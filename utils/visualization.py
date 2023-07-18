import numpy as np
import matplotlib
import math
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show_raw_shape(img, gt, save_path=None, name=None):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(img)
    ax.axis('off')
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(gt, cmap='gray')
    ax.axis('off')

    fig.suptitle('Img, GT', fontsize=10)
    if save_path != None and name != None:
        fig.savefig(save_path + name, dpi=200, bbox_inches='tight')
    ax.cla()
    fig.clf()
    plt.close()


def show_only_seg(img, gt, pre_seg, uncertainty_map,
                  save_path=None, name=None):
    '''
        img + gt + pred_seg + pred_teg + um + reliable_map
    '''
    fig = plt.figure()
    ax = fig.add_subplot(4, 4, 1)
    ax.imshow(img)
    ax.axis('off')
    ax = fig.add_subplot(4, 4, 2)
    ax.imshow(gt, cmap='gray')
    ax.axis('off')

    ax = fig.add_subplot(4, 4, 3)
    ax.imshow(pre_seg, cmap='gray')
    ax.axis('off')

    ax = fig.add_subplot(4, 4, 4)
    ax.imshow(uncertainty_map)
    ax.axis('off')

    fig.suptitle('Img, GT, seg, uncertainty_map', fontsize=10)
    if save_path != None and name != None:
        fig.savefig(save_path + name, dpi=200, bbox_inches='tight')
    ax.cla()
    fig.clf()
    plt.close()


def show_img_gt_segList(img, gt, seg_list, save_path=None, name=None):
    fig = plt.figure()
    ax = fig.add_subplot(4, 3, 1)
    ax.imshow(img)
    ax.axis('off')
    ax = fig.add_subplot(4, 3, 2)
    ax.imshow(gt, cmap='gray')
    ax.axis('off')

    for i in range(4):
        ax = fig.add_subplot(4, 3, 3 + 3 * i)
        ax.imshow(seg_list[i], cmap='gray')
        ax.axis('off')

    fig.suptitle('Img, GT, pre_seg_layer', fontsize=10)
    if save_path != None and name != None:
        fig.savefig(save_path + name, dpi=200, bbox_inches='tight')
    ax.cla()
    fig.clf()
    plt.close()


def show_img_gt_segList_mut_un(img, gt, seg_list, mut_alpha, mut_un, save_path=None, name=None):
    fig = plt.figure()
    ax = fig.add_subplot(4, 5, 1)
    ax.imshow(img)
    ax.axis('off')
    ax = fig.add_subplot(4, 5, 2)
    ax.imshow(gt, cmap='gray')
    ax.axis('off')

    for i in range(4):
        ax = fig.add_subplot(4, 5, 3 + 5 * i)
        ax.imshow(seg_list[i], cmap='gray')
        ax.axis('off')

    ax = fig.add_subplot(4, 5, 4)
    ax.imshow(mut_alpha, cmap='gray')
    ax.axis('off')

    ax = fig.add_subplot(4, 5, 5)
    ax.imshow(mut_un)
    ax.axis('off')

    fig.suptitle('Img, GT, pre_seg_layer, mut_alpha, mut_uncertainty', fontsize=10)
    if save_path != None and name != None:
        fig.savefig(save_path + name, dpi=150, bbox_inches='tight')
    ax.cla()
    fig.clf()
    plt.close()


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
