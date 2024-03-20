# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 16:49
# @Author  : Karry Ren

""" Config file. Set all hyperparameters here. """

# ************************************************************************************ #
# ******************************** FOR I-SPY1 DATASET ******************************** #
# ************************************************************************************ #

# ---- ENVIRONMENT ---- #
GPU_NUM_NETFINAL_BREAST = '1, 2'
GPU_NUM_NETFINAL_REFUGE = '1'  # only mut '0, 1' ||  + ug : 0, 1 || final '0, 1, 2, 3'

# ---- data ---- #
# - Breast
BREAST_BATCH_SIZE = 14
BREAST_TRAIN_NUM = 8200
BREAST_DATA_FILE = r'E:\renkai\Joint_RK\Data\Breast_Dataset\Target_Data'
BREAST_FOLD_FILE = r'E:\renkai\Joint_RK\Data\Breast_Dataset\Data_Fold'
BREAST_FOLD_NUM = '5'

# - Refuge
REFUGE_BATCH_SIZE = 4
REFUGE_TRAIN_NUM = 400
REFUGE_DATA_FILE = r'E:\renkai\Joint_RK\Data\REFUGE_Target_Data'

# ---- MODEL ZOO PATH ---- #
PRETRAIN_RES2NET = r'..\model_zoo\res2net50_v1b_26w_4s-3cf99910.pth'
PRETRAIN_VGG = r'..\model_zoo\5stages_vgg16_bn-6c64b313.pth'

# ---- SAVE RESULT---- #
SAVE_FILE = 'save/'
SAVE_PATH = 'save/checkpoints/'
LOG_PATH = 'save/logs/'
MODEL_SAVE_PATH = 'save/checkpoints/Model_results/'
SEG_RESULT_PATH = 'save/checkpoints/Seg_results/'
SEG_RESULT_PATH_OF_TRAIN = 'save/checkpoints/Seg_results/train/'
SEG_RESULT_PATH_OF_VALID = 'save/checkpoints/Seg_results/valid/'
SEG_RESULT_PATH_OF_TEST = 'test/'

# ---- TRAINING ---- #
NUM_CLASSES_CLS = 2
NUM_CLASSES_SEG = 2
EPOCH = 1

# -- training - lr
LEARNING_RATE = 0.00001
POWER = 0.9

# - BREAST steps
BREAST_STEPS = (BREAST_TRAIN_NUM / BREAST_BATCH_SIZE) * EPOCH
BREAST_BASE_STEPS_OF_30_EPOCHS = (BREAST_TRAIN_NUM / BREAST_BATCH_SIZE) * 30 - 1

# - REFUGE steps
REFUGE_STEPS = (REFUGE_TRAIN_NUM / REFUGE_BATCH_SIZE) * EPOCH
REFUGE_BASE_STEPS_OF_30_EPOCHS = (REFUGE_TRAIN_NUM / REFUGE_BATCH_SIZE) * 30 - 1
WEIGHT_DECAY = 1e-5

# -- training - loss
CLS_ANNEALING_EPOCH = 50
SEG_ANNEALING_EPOCH = 50
CLS_LOSS_WEIGHT = 0.5
MEG_LOSS_WEIGHT = 0.1
SEG_LOSS_WEIGHT = 0.4

# ---- VALID ---- #
VALID_EPOCH = 0
VERBOSE = True
COLOR = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'purple', 'pink', 'peru']
