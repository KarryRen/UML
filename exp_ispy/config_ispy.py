# -*- coding: utf-8 -*-
# @Time    : 2023/3/22 15:45
# @Author  : Karry Ren

""" Config file. Set all hyperparameters for ISPY Dataset. """

# ************************************************************************************ #
# ************************************ GPU SETTINGS ********************************** #
# ************************************************************************************ #
GPU_ID = "0, 1, 2"  # The VISIBLE_DEVICES
GPU_TEST = "0"

# ************************************************************************************ #
# ****************************** DIRECTORY PATH SETTINGS ***************************** #
# ************************************************************************************ #
PATH_SAVE = "save/"
PATH_CHECKPOINTS = f"{PATH_SAVE}/checkpoints/"
PATH_LOGFILE = f"{PATH_SAVE}/logs/"
PATH_MODEL = f"{PATH_CHECKPOINTS}/Model_results/"
SEG_RESULT_PATH_OF_TEST = "test/"

# ************************************************************************************ #
# ********************************** DATASET SETTINGS ******************************** #
# ************************************************************************************ #
ISPY_ROOT_PATH = "E:/renkai/Joint_RK/Code/Data/I-SPY1/ISPY_Dataset_10"
BATCH_SIZE = 8

# ************************************************************************************ #
# *********************************** MODEL SETTINGS ********************************* #
# ************************************************************************************ #
# for model structure
PRETRAINED_RES2NET_PATH = ("E:/renkai/Joint_RK/Code/UML/models/"
                           "model_lib/pretrained_model_zoo/res2net50_v1b_26w_4s-3cf99910.pth")
SEG_CLASS = 2
CLS_CLASS = 2
# for optimizer
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

# ************************************************************************************ #
# ********************************** TRAINING SETTINGS ******************************* #
# ************************************************************************************ #
EPOCH = 100
TRAIN_NUM = 1270
# steps
WHOLE_STEPS = (TRAIN_NUM / BATCH_SIZE) * EPOCH
CHANGE_EPOCH = 30
POWER = 0.9
BASE_STEPS = (TRAIN_NUM / BATCH_SIZE) * CHANGE_EPOCH - 1
# loss
CLS_ANNEALING_EPOCH = 50
SEG_ANNEALING_EPOCH = 50
# loss weight
CLS_LOSS_WEIGHT = 0.5
MUT_LOSS_WEIGHT = 0.1
SEG_LOSS_WEIGHT = 0.4

# ************************************************************************************ #
# ********************************** VALID SETTINGS ********************************** #
# ************************************************************************************ #
VALID_EPOCH = 80

# ************************************************************************************ #
# ********************************** OTHER SETTINGS ********************************** #
# ************************************************************************************ #
COLOR = ["red", "green", "blue", "black", "yellow", "orange", "purple", "pink", "peru"]
