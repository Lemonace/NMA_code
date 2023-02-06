# coding: utf-8 -*-
import shutil
import time
import os

# result
RANDOM_SEED = 2021
BATCH_SIZE = 1024
IMP_LOSS_WEIGHT = 0.02
# basic config
EPOCH = 1 
LEARNING_RATE = 0.005
DATA_MODE = 1 # 1:local train，2:local test, 3:docker evaluate
MODEL_NAME = "listwise_clpm_model_b"
# poi类别特征
FEATURE_CATE_NUM = 6 # v1r3:19
# dense特征
FEATURE_DENSE_NUM = 5  # v1:28 v1r2:79 v1r3:83
# 预估值特征
FEATURE_CXR_NUM = 3
# 环境特征
FEATURE_ENV_NUM = 2
# 自然poi

# N: Cut Number of POI For Train
POI_NUM = 5
AD_POI_NUM = 2
NATURE_POI_NUM = 3
FEATURE_NATURE_POI = 3
FEATURE_NUM = 9

# 属性特征：KA AOR BRAND
FEATURE_ATTR_NUM = 3

# DELIVERY_FEAT
DELIVERY_FEAT_NUM = 4
MOD_BASE = 4194300
# OUT NUM
OUT_NUM = 5

PLACE_HOLDER_NUM = 11
DENSE_FEAT_NUM = 439


# embedding_look_up维度
CATE_FEATURE_EMBEDDINGS_SHAPE = [1 << 22, 8]

# 网络结构参数
MODEL_PARAMS = {
    'INPUT_TENSOR_LAYERS_A': [60, 32, 10],
    'INPUT_TENSOR_LAYERS_B': [50, 20],
    'INPUT_TENSOR_LAYERS_C': [50, 20],
    'INPUT_TENSOR_LAYERS_D': [50, 20],
    'INPUT_TENSOR_LAYERS_E': [50, 20]
}
A_INPUT_DIM = POI_NUM * (MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1])

DIN_CONF = {}

# checkout-point 参数

# 网络结构参数
LIST_MODEL_PARAMS = {
    'INPUT_TENSOR_LAYERS_A': [60, 32, 10],
    'INPUT_TENSOR_LAYERS_B': [50, 20],
    'INPUT_TENSOR_LAYERS_C': [50, 20],
    'INPUT_TENSOR_LAYERS_D': [50, 20],
    'INPUT_TENSOR_LAYERS_E': [50, 20]
}
CATE_FEATURE_EMBEDDINGS_SHAPE = [1 << 22, 8]
LIST_MODEL_POI_NUM = 5 # 2ad+3nature
LIST_MODEL_CATE_FEATURE_NUM = 6
LIST_MODEL_INPUT_DIM = LIST_MODEL_POI_NUM * (LIST_MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1)
# LIST_MODEL_CHECKPOINT_PATH = '/Users/lemonace/workspace/floating_ad_rl/model/point_model_base/model.ckpt-512'
LIST_MODEL_CHECKPOINT_PATH = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/floating_ad_rl/model/point_model_base/model.ckpt-20449'



# train data
# /users/lemonace/Downloads/tfrecord-rl-limit5-v1
if DATA_MODE == 1:
    TRAIN_FILE = ['/users/lemonace/Downloads/docker_data/nma_data/part-r-00000']
    VALID_FILE = TRAIN_FILE
    PREDICT_FILE = VALID_FILE
    TEST_FILE = PREDICT_FILE
elif DATA_MODE == 2:
    TRAIN_FILE = ['test_data/part-r-*']
    VALID_FILE = TRAIN_FILE
    TEST_FILE = VALID_FILE
elif DATA_MODE == 3:
    TRAIN_FILE = ["train_data/part-r-*"]
    VALID_FILE = ["test_data/part-r-*"]
    TEST_FILE= ["test_data/part-r-*"]

# 辅助脚本
MEAN_VAR_PATH_POI = "./avg_std/poi"
MEAN_VAR_PATH_DELIVERY = "./avg_std/delivery"
MODEL_SAVE_PATH = "../model/" + MODEL_NAME
MODEL_SAVE_PB_EPOCH_ON = False
MODEL_SAVE_PB_EPOCH_PATH = MODEL_SAVE_PATH + "_pbs"
