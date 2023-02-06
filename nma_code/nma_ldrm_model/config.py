# coding: utf-8 -*-
import shutil
import time
import os


POI_FEA_NUM = 8
POI_NUM = 10
SUB_POI_NUM = 3
INPUT_FEA_NUM = 18
# basic config
EPOCH = 10
BATCH_SIZE = 1024
RANDOM_SEED = 2022
LEARNING_RATE = 1e-4
DATA_MODE = 1 # 1:local train，2:test, 3:docker evaluate
MODEL_NAME = "nma_ldrm_model"

MAX_LIST_NUM = 90
PAY_LIST_NUM = 1

JFB_BASE = 0.93
JFB_K = 4 
CE_K = 0 
CE_VALUE_K = 10 
SW_K = 0
PAY_K = 0
PAY_VALUE_K = 0
RS_K1 = 0

GMVK = 0
GMVK_BASE = 0

# feature settings
PER_LIST_POI_NUM = 2
POI_PREDICT_FEA_NUM = 8 
POI_DENSE_FEA_NUM = 5 
POI_CATE_FEA_NUM = 6 
POI_MASK_NUM = MAX_LIST_NUM
CATE_EMBEDDING_SIZE = 4
CATE_VOCAB_SIZE = (1 << 22)
INPUT_DIM = POI_PREDICT_FEA_NUM + POI_DENSE_FEA_NUM + POI_CATE_FEA_NUM * CATE_EMBEDDING_SIZE
# label settings
LABEL_LEN = 0 
TOP_K = 1

# 网络结构参数
MODEL_PARAMS = {
    'INPUT_TENSOR_LAYERS_A': [50, 10],
    'INPUT_TENSOR_LAYERS_B': [32, 16]
}

# embedding_look_up维度
CATE_FEATURE_EMBEDDINGS_SHAPE = [1 << 22, 8]

# network parameters
USE_PRE_LIST_WISE_MODEL = False
MU_INPUT_DIM = INPUT_DIM
LAMBDA_INPUT_DIM = INPUT_DIM * PER_LIST_POI_NUM
MU_MLP = [32, 8, 1]
LAMBDA_MLP = [32, 8, 1]

# reward parameters: ctr*pay, ctr*bid, gmv
REWARD_WEIGHT = [0.0, 0.1, 0.0]

# Loss weights: pay*ctr+ctr*bid+gmv, aux_ce, aux_payrate
LOSS_WEIGHT = [0.2, 1.0, 1.0]

# use pre-trained list-wise model
USE_PRE_TRAINED_LIST_WISE_MODEL = True
LWP = {}    # list-wise params
if DATA_MODE == 1:
    LWP['CHECKPOINT_PATH'] = '../../data/yf_epcsf/model.ckpt-1572540'
else:
    LWP['CHECKPOINT_PATH'] = '/home/hadoop-hmart-waimaiad/cephfs/data/wuxiaoxu04/projects/deep_rank/model_ctr_calibration/yf_epcsf/model.ckpt-1572540'
LWP['LIMIT_NUM'] = 4
LWP['POI_CATE_FEA_NUM'] = 6
LWP['CATE_EMBEDDING_SIZE'] = 4
LWP['POI_DENSE_FEA_NUM'] = 5
LWP['POI_EMBEDDING_FEA_NUM'] = 0
LWP['P_NETWORK_INPUT_DIM'] = LWP['POI_CATE_FEA_NUM'] * LWP['CATE_EMBEDDING_SIZE'] + LWP['POI_DENSE_FEA_NUM'] + LWP[
    'POI_EMBEDDING_FEA_NUM']
LWP['POI_PREDICT_FEA_NUM'] = 3
LWP['POI_P_PREDICT_OUT_NUM'] = 2
LWP['P_NETWORK_DEEP_LAYERS'] = "64,32,%d" % LWP['POI_P_PREDICT_OUT_NUM']

LWP['DENSE_FEA_NUM'] = 2
LWP['EMBEDDING_FEA_NUM'] = 0
LWP['CATE_FEA_NUM'] = 0
LWP['A_NETWORK_DEEP_LAYERS'] = "50,20"
LWP['A_NETWORK_INPUT_DIM'] = LWP['DENSE_FEA_NUM'] + LWP['EMBEDDING_FEA_NUM'] + LWP['LIMIT_NUM'] * (
            LWP['POI_PREDICT_FEA_NUM'] + LWP['POI_P_PREDICT_OUT_NUM']) + LWP['CATE_FEA_NUM']
LWP['MEAN_VAR_PATH'] = './avg_std/avg_std_ctr_cal_poi_dense'

# use position weight
USE_BIAS_WEIGHT = True

# train data
if DATA_MODE == 1:
    TRAIN_FILE = ["avito_v1_simulate_for_DNA/train_data/part-r-00001"]
    VALID_FILE = ["avito_v1_simulate_for_DNA/train_data/part-r-00041"]
else:
    TRAIN_FILE = ["avito_v1_simulate_for_DNA/train_data/part-r-000[0-3]*"]
    VALID_FILE = ["avito_v1_simulate_for_DNA/train_data/part-r-0004*"]

# 辅助脚本
MEAN_VAR_PATH_POI = "./avg_std/poi"
MODEL_SAVE_PATH = "../model/" + MODEL_NAME
MODEL_SAVE_PB_EPOCH_ON = False
MODEL_SAVE_PB_EPOCH_PATH = MODEL_SAVE_PATH + "_pbs"
