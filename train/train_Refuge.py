import torch.backends.cudnn as cudnn
import os
import warnings
from models.UMLNet import UMLNet
import config
from models.loss.cls_loss import *
from models.loss.seg_loss import *
from dataset.my_refuge_datasets import *
import time
from utils.tool_functions import *
from utils.eval_functions import *
from utils.visualization import draw_curves

# ---- ignore all warnings ---- #
warnings.filterwarnings("ignore")

# ---- Setting a argparse and add var ---- #
model_urls = {}  # Model pre train urls


def SOTA_EXPERIMENT_UML_REFUGE():
    """ Final Network UML """

    # ---- Setting cuda environment ---- #
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM_NETFINAL_REFUGE
    cudnn.enabled = True
    cudnn.benchmark = True

    # ---- Load training and validation data ---- #
    train_loader = torch.utils.data.DataLoader(
        JointRefugeDataset(path_refuge_data=config.REFUGE_DATA_FILE, data_type='train'),
        batch_size=config.REFUGE_BATCH_SIZE, num_workers=0, shuffle=True, pin_memory=False, drop_last=True)
    train_test_loader = torch.utils.data.DataLoader(
        JointRefugeDataset(path_refuge_data=config.REFUGE_DATA_FILE, data_type='train'),
        batch_size=1, num_workers=0, shuffle=False, pin_memory=False, drop_last=True)
    print("train image num:", len(train_test_loader))
    valid_loader = torch.utils.data.DataLoader(
        JointRefugeDataset(path_refuge_data=config.REFUGE_DATA_FILE, data_type='valid'),
        batch_size=1, num_workers=0, shuffle=False, pin_memory=False, drop_last=True)
    print("valid image num:", len(valid_loader))

    # ---- Create NetWork, Optimizer and Loss List --- #
    model = UMLNet(config)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # ---- Loss List ---- #
    sum_loss_list_all = []
    cls_loss_list_all = []
    seg_loss_list_all = []
    mut_loss_list_all = []

    # ---- valid settings the index list of cls, seg, prob_seg in train or valid ---- #
    CLS_SEG_train = []
    CLS_SEG_valid = []

    # ---- other settings prepare save path ---- #
    if not os.path.isdir(config.SAVE_FILE):
        os.mkdir(config.SAVE_FILE)
    if not os.path.isdir(config.SAVE_PATH):
        os.mkdir(config.SAVE_PATH)
    if not os.path.isdir(config.LOG_PATH):
        os.mkdir(config.LOG_PATH)
    if not os.path.isdir(config.MODEL_SAVE_PATH):
        os.mkdir(config.MODEL_SAVE_PATH)
    if not os.path.isdir(config.SEG_RESULT_PATH):
        os.mkdir(config.SEG_RESULT_PATH)
    if not os.path.isdir(config.SEG_RESULT_PATH_OF_TRAIN):  # train seg demo
        os.mkdir(config.SEG_RESULT_PATH_OF_TRAIN)
    if not os.path.isdir(config.SEG_RESULT_PATH_OF_VALID):  # valid seg demo
        os.mkdir(config.SEG_RESULT_PATH_OF_VALID)
    f_path = config.LOG_PATH + 'training_output.log'
    logfile = open(f_path, 'a')

    # ---- Print Information ---- #
    print_f(os.getcwd(), f=logfile)
    print_f('=== Device: {} === '.format(os.environ['CUDA_VISIBLE_DEVICES']), f=logfile)
    print_f('=== Time: {} ==='.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), f=logfile)
    print_f('=== Settings ===', f=logfile)
    print_f('==| Initial LR: {} |=='.format(config.LEARNING_RATE), f=logfile)

    # ---- Train && Valid ---- #
    for epoch in range(config.EPOCH):
        # ---- loss list ---- #
        sum_loss_one_epoch = []
        cls_loss_one_epoch = []  # loss of cls
        seg_loss_one_epoch = []  # loss of seg
        mut_loss_one_epoch = []  # loss of mut

        model.train()

        # ---- start the training ---- #
        for i_iter, batch in enumerate(train_loader):
            step = (config.REFUGE_TRAIN_NUM / config.REFUGE_BATCH_SIZE) * epoch + i_iter
            images, masks, labels, image_name = batch
            images = images.cuda()
            labels = labels.cuda()
            masks = masks.unsqueeze(1).cuda()
            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, step, epoch, whole_steps=config.REFUGE_STEPS,
                                      base_steps=config.REFUGE_BASE_STEPS_OF_30_EPOCHS)

            cls_alpha, cls_uncertainty, \
            mutual_evidence, mutual_alpha, mutual_uncertainty, \
            seg_list, seg_list_view, reliable_mask_list = model(images)

            # --- cls loss
            cls_loss = cls_evidence_loss(labels, cls_alpha, config.NUM_CLASSES_CLS, epoch, config.CLS_ANNEALING_EPOCH)
            # --- seg loss
            seg_loss = seg_dice_loss(seg_list[0], masks, c=config.NUM_CLASSES_SEG)
            for i in range(1, len(seg_list)):
                seg_loss += seg_dice_loss(seg_list[i], masks, c=config.NUM_CLASSES_SEG)
            seg_loss = seg_loss / 4.0
            # --- meg loss
            mut_loss = seg_evidence_loss(masks, mutual_alpha,
                                         config.NUM_CLASSES_SEG, epoch,
                                         config.CLS_ANNEALING_EPOCH,
                                         mutual_evidence)

            # --- sum loss and back
            sum_loss = config.CLS_LOSS_WEIGHT * cls_loss + config.SEG_LOSS_WEIGHT * seg_loss + config.MEG_LOSS_WEIGHT * mut_loss
            sum_loss.backward()
            optimizer.step()

            # list the loss for one epoch
            sum_loss_one_epoch.append(sum_loss.cpu().data.numpy())
            cls_loss_one_epoch.append(cls_loss.cpu().data.numpy())
            seg_loss_one_epoch.append(seg_loss.cpu().data.numpy())
            mut_loss_one_epoch.append(mut_loss.cpu().data.numpy())

        # ---- train log ---- #
        line = \
            "Train-Epoch [%d/%d] [All]: " \
            "Loss = %.6f, " \
            "Cls_loss = %.6f, " \
            "Seg_loss = %.6f, " \
            "Mut_loss = %.6f, " \
            "LR = %0.9f" % (
                epoch, config.EPOCH,
                np.nanmean(sum_loss_one_epoch),
                np.nanmean(cls_loss_one_epoch),
                np.nanmean(seg_loss_one_epoch),
                np.nanmean(mut_loss_one_epoch),
                lr)

        sum_loss_list_all.append(np.nanmean(sum_loss_one_epoch))
        cls_loss_list_all.append(np.nanmean(cls_loss_one_epoch))
        seg_loss_list_all.append(np.nanmean(seg_loss_one_epoch))
        mut_loss_list_all.append(np.nanmean(mut_loss_one_epoch))
        print_f(line, f=logfile)

        # ---- save model and do validation ---- %
        if epoch >= config.VALID_EPOCH:
            # ---- save model ---- #
            filename = str(epoch) + '_uml_refuge.pt'
            torch.save(model.module.state_dict(), os.path.join(config.MODEL_SAVE_PATH, filename))

            # ---- eval train loader ----#
            train_test_result = eval_UML_REFUGE(valid_data_loader=train_test_loader,
                                                model=model,
                                                epoch=epoch,
                                                save_seg_path=config.SEG_RESULT_PATH_OF_TRAIN,
                                                verbose=config.VERBOSE,
                                                whole_epoch=config.EPOCH)
            # cls
            [ACC, F1] = train_test_result['cls']
            line = "Train-EVAL-EPOCH [%d/%d] [Cls]: ACC = %f, F1 = %f" % (epoch, config.EPOCH, ACC, F1)
            print_f(line, f=logfile)
            # seg
            [DICE1, ASSD1, DICE2, ASSD2] = train_test_result['seg']
            line = "Train-EVAL-EPOCH [%d/%d] [Seg]: DICE1 = %f, ASSD1 = %f, DICE2 = %f, ASSD2 = %f" % (
                epoch, config.EPOCH, DICE1, ASSD1, DICE2, ASSD2)
            print_f(line, f=logfile)
            # cls seg two task index #
            CLS_SEG_train.append((ACC + F1 + DICE1 + DICE2) / 4)
            print("---------------------------------------------------------------------------")

            # ---- eval the valid loader ---- #
            valid_result = eval_UML_REFUGE(valid_data_loader=valid_loader,
                                           model=model,
                                           epoch=epoch,
                                           save_seg_path=config.SEG_RESULT_PATH_OF_VALID,
                                           verbose=config.VERBOSE,
                                           whole_epoch=config.EPOCH)
            # cls #
            [ACC, F1] = valid_result['cls']
            line = "Valid-EVAL-EPOCH [%d/%d] [Cls]: ACC = %f, F1 = %f" % (epoch, config.EPOCH, ACC, F1)
            print_f(line, f=logfile)
            # seg #
            [DICE1, ASSD1, DICE2, ASSD2] = valid_result['seg']
            line = "Valid-EVAL-EPOCH [%d/%d] [Seg]: DICE1 = %f, ASSD1 = %f, DICE2 = %f, ASSD2 = %f " % (
                epoch, config.EPOCH, DICE1, ASSD1, DICE2, ASSD2)
            print_f(line, f=logfile)
            # cls seg two task index #
            CLS_SEG_valid.append((ACC + F1 + DICE1 + DICE2) / 4)
            print("***************************************************************************\n")
            print()

    # ---- Plot Loss curve ---- #
    filename = os.path.join(config.LOG_PATH, 'loss_curves.png')
    data_list = [sum_loss_list_all, cls_loss_list_all, seg_loss_list_all]
    label_list = ['sum_loss', 'cls_loss', 'seg_loss']
    draw_curves(data_list=data_list, label_list=label_list, color_list=config.COLOR[0:3], filename=filename)
    print("---------------------------------------------------------------------------")

    # ---- Select the best Epoch ---- #
    index_of_best_train = CLS_SEG_train.index(max(CLS_SEG_train))
    line = "[TRAIN Best Performance index %d]" % (index_of_best_train + config.VALID_EPOCH)
    print_f(line, f=logfile)

    index_of_best_valid = CLS_SEG_valid.index(max(CLS_SEG_valid))
    line = "[VALID Best Performance index %d]" % (index_of_best_valid + config.VALID_EPOCH)
    print_f(line, f=logfile)


if __name__ == '__main__':
    SOTA_EXPERIMENT_UML_REFUGE()
