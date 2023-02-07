# -*- coding: utf-8 -*-
import numpy as np

from config import *
from model import *


def create_estimator():
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        tf_random_seed=RANDOM_SEED,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        model_dir=MODEL_SAVE_PATH,
        keep_checkpoint_max=2,
        log_step_count_steps=1000,
        session_config=session_config)
    nn_model = DNN()
    estimator = tf.estimator.Estimator(model_fn=nn_model.model_fn_estimator, config=config)

    return estimator, nn_model



def save_model_pb_with_estimator(estimator, params, export_dir_base):
    estimator._params['save_model'] = params['save_model']

    def _serving_input_receiver_fn():
        receiver_tensors = {
            # ctr cvr gmv预估值 && bid
            'screen_predict_feature': tf.placeholder(tf.float32, [None, MAX_LIST_NUM,  PER_LIST_POI_NUM * POI_PREDICT_FEA_NUM],
                                                name='screen_predict_feature'),
            # dense 特征 (价格，评分)
            'screen_dense_feature': tf.placeholder(tf.float32, [None, MAX_LIST_NUM, PER_LIST_POI_NUM * POI_DENSE_FEA_NUM],
                                                name='screen_dense_feature'),
            # 离散特征(品类)
            'screen_cate_feature': tf.placeholder(tf.int64, [None, MAX_LIST_NUM, PER_LIST_POI_NUM * POI_CATE_FEA_NUM],
                                                name='screen_cate_feature'),
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)

    export_dir = estimator.export_saved_model(export_dir_base=export_dir_base,
                                              serving_input_receiver_fn=_serving_input_receiver_fn)
    estimator._params.pop('save_model')
    return export_dir.decode()



def calculate_result(result_generator, epcho):
    VVCA_BID = 0
    VVCA_CHARGE = 0
    VVCA_CTR = 0
    VVCA_CPM = 0
    VVCA_SW = 0
    VVCA_GMV = 0
    VCG_CTR = 0
    VCG_CPM = 0
    VCG_SW = 0
    VCG_GMV = 0
    UGSP_BID = 0
    UGSP_CHARGE = 0
    UGSP_CTR = 0
    UGSP_CPM = 0
    UGSP_SW = 0
    UGSP_GMV = 0
    count = 0
    for result in result_generator:
        '''
        print(result['VVCA_BID'])
        print(result['VVCA_CHARGE'])
        print(result['VVCA_CTR'])
        print(result['VVCA_CPM'])
        print(result['UGSP_CHARGE'])
        '''
        count += len(result['VVCA_CTR'])
        VVCA_BID += np.sum(result['VVCA_BID'])
        VVCA_CHARGE += np.sum(result['VVCA_CHARGE'])
        VVCA_CTR += np.sum(result['VVCA_CTR'])
        VVCA_CPM += np.sum(result['VVCA_CPM'])
        VVCA_SW += np.sum(result['VVCA_SW'])
        VVCA_GMV += np.sum(result['VVCA_GMV'])
        VCG_CTR += np.sum(result['VCG_CTR'])
        VCG_CPM += np.sum(result['VCG_CPM'])
        VCG_SW += np.sum(result['VCG_SW'])
        VCG_GMV += np.sum(result['VCG_GMV'])
        UGSP_BID += np.sum(result['UGSP_BID'])
        UGSP_CHARGE += np.sum(result['UGSP_CHARGE'])
        UGSP_CTR += np.sum(result['UGSP_CTR'])
        UGSP_CPM += np.sum(result['UGSP_CPM'])
        UGSP_SW += np.sum(result['UGSP_SW'])
        UGSP_GMV += np.sum(result['UGSP_GMV'])
    print("NMA_EPCHO:{}".format(epcho))
    print("NMA_CTR:{}".format(VVCA_CTR/count/2))
    print("NMA_CPM:{}".format(VVCA_CPM/count/2))
    print("NMA_SW:{}".format(VVCA_SW/count/2))

if __name__ == '__main__':

    estimator, nn_model = create_estimator()

    with tick_tock("DATA_INPUT") as _:
        #valid_input_fn = input_fn_maker(VALID_FILE, False, batch_size=BATCH_SIZE, epoch=1)
        valid_input_fn = input_fn_maker(VALID_FILE, False, batch_size=10240, epoch=1)
    count = 0
    if DATA_MODE == 1:
        for i in range(EPOCH):
            for idx, data in enumerate(TRAIN_FILE):
                with tick_tock("DATA_INPUT") as _:
                    train_input_fn = input_fn_maker([data], True, batch_size=BATCH_SIZE, epoch=1)
                with tick_tock("TRAIN") as _:
                    estimator.train(train_input_fn)
                with tick_tock("valid") as _:
                    result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
                    calculate_result(result_generator, count)
                count += 1
    if DATA_MODE == 2:
        with tick_tock("valid") as _:
            if count == 0:
                result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
                calculate_result(result_generator, count)
