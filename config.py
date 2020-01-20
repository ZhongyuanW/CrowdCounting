# -*- coding: utf-8 -*-
# @Time    : 10/14/19 12:39 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : config.py
# @Software: PyCharm

HOME = "/home/zhongyuan/datasets/ShanghaiTech"
DATASET = "part_A_final"

CUDA = "6"

RESUME = False

BATCH_SIZE = 1
MOMENTUM = 0.95
WEIGHT_DECAY = 5*1e-4
LEARNING_RATE = 1e-7
MAX_EPOCH = 2000
STEPS = (i for i in range(MAX_EPOCH))
LR_DECAY = None
OPTIMIZER = "SGD"
LOSS_F = "MSE"

MODEL = "csrnet"

SAVE_PATH = "%s/weights/baseline/%s%s_batch%s"%(MODEL,OPTIMIZER,str(LEARNING_RATE),str(BATCH_SIZE))

RANDOM_CROP = False
RANDOM_FLIP = False        # 0.5
RANDOM_2GRAY = False       # 0.2
DIVIDE = False             # 16









VAR_LIST = ["BATCH: %d"%BATCH_SIZE, "OPTIM: %s"%OPTIMIZER, "LR: %s"%str(LEARNING_RATE), "LOSS_F: %s"%LOSS_F,
            "CUDA: %s"%CUDA,"LR_DECAY: %s"%LR_DECAY, "MODEL: %s"%MODEL, "RANDOM_CROP: %s"%str(RANDOM_CROP),
            "RANDOM_FLIP: %s"%str(RANDOM_FLIP),"RANDOM_2GRAY: %s"%str(RANDOM_2GRAY),"DIVIDE: %s"%DIVIDE,
            "SAVE_PATH: %s"%SAVE_PATH, "DATASET: %s"%DATASET]
