# coding: utf-8 -*-
from data_input import *
from tools import *

import numpy as np
import tensorflow as tf


class DNN:
    def __init__(self):
        feature_mask_list, parse_mask_flag, feature_hold_cnt = parse_mask_file("codes/poi_feature_mask")
        if parse_mask_flag & (len(feature_mask_list) == INPUT_DIM):
            self.feature_mask = np.nonzero(feature_mask_list)[0].tolist()
            self.feature_mask_on = True
        else:
            self.feature_mask_on = False

    def _create_weights(self):
        with tf.name_scope('feature_emb_weights'):
            self.feature_weights = {
                'embedding': tf.get_variable('embedding',
                                             shape=[CATE_VOCAB_SIZE, LWP['CATE_EMBEDDING_SIZE']],
                                             initializer=tf.zeros_initializer())
            }

    def _init_feature(self, features):
        # reshape input
        self.list_poi_predict_feature = features['list_poi_predict_feature']
        self.list_poi_dense_feature = features['list_poi_dense_feature']
        self.list_poi_mask_feature = features['list_poi_mask_feature']
        self.list_poi_cate_idx = features['list_poi_cate_feature']
        self.list_size = tf.minimum(features['list_size'], MAX_LIST_NUM)
        self.list_poi_cate_feature = tf.reshape(
            tf.nn.embedding_lookup(self.feature_weights['embedding'], self.list_poi_cate_idx),
            [-1, MAX_LIST_NUM, PER_LIST_POI_NUM, POI_CATE_FEA_NUM * CATE_EMBEDDING_SIZE])
        # split feature   [bs, list_size, poi_size]
        self.list_poi_bid = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(0, 1)), axis=-1), axis=-1)
        self.list_poi_ctr = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(1, 2)), axis=-1), axis=-1)
        self.list_poi_cvr = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(2, 3)), axis=-1), axis=-1)
        self.list_poi_gmv = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(3, 4)), axis=-1), axis=-1)
        zero_tmp = tf.zeros_like(self.list_poi_ctr)
        self.list_poi_pprice = tf.where(tf.greater(self.list_poi_ctr * self.list_poi_cvr, zero_tmp),
                                        tf.div(self.list_poi_gmv, self.list_poi_ctr * self.list_poi_cvr), zero_tmp)
        # build mask & avg_feature
        self.list_len_mask = tf.squeeze(tf.sequence_mask(lengths=self.list_size, maxlen=MAX_LIST_NUM, dtype=tf.float32), axis=1)
        # self.list_poi_feature = tf.concat([self.list_poi_dense_feature, self.list_poi_cate_feature], axis=-1)
        self.list_poi_feature = tf.concat([self.list_poi_predict_feature, self.list_poi_cate_feature], axis=-1)
        self.list_poi_feature = self.tfPrint(self.list_poi_feature, "list_poi_feature")
        # feature mask
        if self.feature_mask_on:
            self.list_poi_feature = tf.gather(self.list_poi_feature, self.feature_mask, axis=-1)

    def _create_rankscore_function(self, bid, ctr, gmv, rs_mu, rs_lambda, K=1):
        #output = bid * ctr * rs_mu + rs_lambda  # [bs, list_len, poi_len, 1]
        output = bid * ctr * rs_mu + rs_lambda  # [bs, list_len, poi_len]
        output = tf.reduce_sum(output, axis=2)  # [bs, list_len]
        top_list_idx = tf.math.top_k(output, k=TOP_K).indices  # [-1, K]
        return output, top_list_idx

    def _create_differentiable_sorting(self, input):
        N = MAX_LIST_NUM
        Ar1 = tf.reshape(tf.tile(input, [1, N]), [-1, N, N])
        Ar2 = tf.transpose(Ar1, perm=[0, 2, 1])
        Ar = tf.abs(Ar1 - Ar2)
        ArOne = tf.reduce_sum(Ar, axis=2)
        k = tf.range(start=1, limit=N + 1)
        W = tf.tile(tf.reshape(N + 1 - 2 * k, [N, 1]), [1, N])
        ArOne_matrix = tf.tile(tf.reshape(ArOne, [-1, 1, N]), [1, N, 1])  # (batchsize;N;N)
        input_matrix = tf.tile(tf.reshape(input, [-1, 1, N]), [1, N, 1])  # (batchsize;N;N)
        Ck = tf.multiply(input_matrix, tf.cast(W, dtype=tf.float32)) - ArOne_matrix
        Mr = tf.nn.softmax(Ck, axis=2)
        Mr = self.tfPrint(Mr, "Mr")
        return Mr

    def _create_vvca_payment(self, rs, top_list_idx, list_poi_mask, N):
        self.next_list_rs = tf.tile(tf.reshape(rs, [-1, 1, 1, N]), [1, N, PER_LIST_POI_NUM, 1])  # [-1, N, 4, N]
        # todo : list_poi_mask，对于list2,list3来说，都需要去除前一个，所以目前仅支持第一个list训练，如果需要多个list，需优化mask
        self.masked_next_list_rs = self.next_list_rs * list_poi_mask  # [-1, N, 4, N]
        self.except_self_item_next_max_list_rs = tf.reduce_max(self.masked_next_list_rs, axis=-1)  # [-1, N, 4]

        self.tile_list_rs = tf.tile(tf.reshape(rs, [-1, N, 1]), [1, 1, PER_LIST_POI_NUM])  # [-1, N, 4]
        self.rs_gap = self.tile_list_rs - self.except_self_item_next_max_list_rs  # [-1, N, 4]
        self.pay = self.list_poi_bid - self.rs_gap / (self.list_poi_ctr * self.rs_mu)  # [-1, N, 4]
        # todo : 做一些异常处理，截断
        # 只有第一个计费
        # self.tile_top1_list_idx = tf.tile(tf.reshape(top_list_idx, [-1, 1, 1]), [1, PER_LIST_POI_NUM, 1])  # [-1, 4, 1]
        # self.pay_top1 = tf.batch_gather(tf.transpose(self.pay, perm=[0, 2, 1]), self.tile_top1_list_idx)  # [-1, 4, 1]
        pay = tf.gather(self.pay, self.top_list_idx, axis=1, batch_dims=-1)
        pay = self.tfPrint(pay, "pay")
        return pay

    def _create_model(self, features):
        # output poi_feat = [BATCH_SIZE;LIST_NUM;POI_NUM;FEATURE_NUM]

        self._init_feature(features)
        with tf.name_scope('rs_mu_lambda'):
            # mu与分配无关
            # rs_mu, rs_lambda = [BATCH_SIZE;LIST_NUM;POI_NUM;FEATURE_NUM]
            self.rs_mu = self._create_mu_network(self.list_poi_feature)
            # lambda可以与分配有关，
            # TODO 目前和mu的特征相同，待优化
            self.rs_lambda = self._create_lambda_network(self.list_poi_feature)
            self.rs_mu = self.tfPrint(self.rs_mu, "rs_mu")
            self.rs_lambda = self.tfPrint(self.rs_lambda, "rs_lambda")
        with tf.name_scope('rankscore_function'):
            # vvca_rs [-1, N], [-1, K]
            self.rs, self.top_list_idx = self._create_rankscore_function(self.list_poi_bid, self.list_poi_ctr, self.list_poi_gmv, self.rs_mu, self.rs_lambda)
            self.rs = self.tfPrint(self.rs, "rs")
        with tf.name_scope('vvca_charge'):
            # TODO mask数据需要重写
            self.vvca_charge = self._create_vvca_payment(self.rs, self.top_list_idx, self.list_poi_mask_feature, MAX_LIST_NUM)
        with tf.name_scope('differentiable_sorting'):
            # 多序列做可微分排序
            self.Mr = self._create_differentiable_sorting(self.rs)

    def _create_mu_network(self, list_poi_feat):
        fc_out = list_poi_feat
        for i in range(0, len(MU_MLP)):
            dense_name = "MU_NET_" + str(i)
            fc_out = tf.layers.dense(fc_out, MU_MLP[i], activation=None, name=dense_name)
            fc_out = tf.nn.swish(fc_out)
        fc_out = tf.nn.sigmoid(fc_out)
        return tf.squeeze(fc_out, axis=-1)

    def _create_lambda_network(self, list_poi_feat):
        fc_out = list_poi_feat
        for i in range(0, len(LAMBDA_MLP)):
            dense_name = "LAMBDA_NET_" + str(i)
            fc_out = tf.layers.dense(fc_out, LAMBDA_MLP[i], activation=None, name=dense_name)
            fc_out = tf.nn.swish(fc_out)
        return tf.squeeze(fc_out, axis=-1)

    def _create_loss(self, labels):
        N = MAX_LIST_NUM
        # 主loss分位置加权
        Mr = self.Mr
        Mr = self.tfPrint(Mr, "Mr")
        _, reward_pay_tile = self._gen_reward(N, "pay*ctr")
        _, reward_gmv_tile = self._gen_reward(N, "gmv")
        all_reward, _ = self._gen_reward(N)
        self.loss_pay = -tf.reduce_sum(
            tf.reduce_sum(tf.multiply(Mr, reward_pay_tile), axis=[2, 1]))
        self.loss_gmv = -tf.reduce_sum(
            tf.reduce_sum(tf.multiply(Mr, reward_gmv_tile), axis=[2, 1]))
        self.loss = tf.reduce_mean(self.loss_pay + self.loss_gmv)

    def _gen_reward(self, N, type="all"):
        if type == 'pay*ctr':
            # [BATCH_SIZE;LIST_NUM;LIST_SIZE]
            reward = self.list_poi_bid * self.list_poi_ctr
        elif type == 'bid*ctr':
            reward = self.list_poi_bid * self.list_poi_ctr
        elif type == 'gmv':
            reward = self.list_poi_ctr * self.list_poi_cvr * self.list_poi_pprice
        else:
            reward = self.list_poi_bid * self.list_poi_ctr
        # reward = self.tfPrint(reward, type)
        # reward_tile for Mr
        # self.top_list_idx : [BATCH_SIZE, K]
        reward = tf.reduce_sum(reward, axis=2) # [BATCH_SIZE;LIST_NUM]
        reward = self.tfPrint(reward, "reward")
        updates = tf.ones_like(self.top_list_idx)
        reward_mask = tf.cast(self._batch_scatter_nd_new(self.top_list_idx, updates, TOP_K, tf.shape(reward)), tf.float32)
        reward = reward * reward_mask
        reward_tile = tf.reshape(tf.tile(reward, [1, N]), [-1, N, N])
        reward_tile = self.tfPrint(reward_tile, "reward_tile")
        return reward, reward_tile

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def model_fn_estimator(self, features, labels, mode, params):
        self._create_weights()
        self._create_model(features)

        if mode == tf.estimator.ModeKeys.TRAIN:
            self._create_loss(labels)
            self._create_optimizer()
            self._build_indicator(labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op,
                                              training_hooks=[self.logging_hook])

    def _build_indicator(self, labels):
        # GMV, BID, uGSP_CHARGE, VVCA_CHARGE, JFB
        # labels_result['charge']， self.list_poi_bid, self.list_poi_ctr, self.list_poi_cvr, self.list_poi_pprice
        uGSP_CHARGE = tf.reduce_mean(tf.gather(labels['charge'], self.top_list_idx, axis=1, batch_dims=-1))
        BID = tf.reduce_mean(tf.gather(self.list_poi_bid, self.top_list_idx, axis=1, batch_dims=-1))
        VVCA_CHARGE = tf.reduce_mean(self.vvca_charge)

        def formatter_log(tensors):
            log_string = "indicator_info: step {}:, loss={:.4f}, " \
                          "uGSP_CHARGE={:.4f}, BID={:.4f}, VVCA_CHARGE={:.4f}".format(
                tensors["step"], tensors["loss"], tensors["uGSP_CHARGE"], tensors["BID"], tensors["VVCA_CHARGE"]
            )
            return "\n" + log_string

        self.logging_hook = tf.train.LoggingTensorHook({"loss": self.loss,
                                                        "step": tf.train.get_global_step(),
                                                        "uGSP_CHARGE": uGSP_CHARGE,
                                                        "BID": BID,
                                                        "VVCA_CHARGE": VVCA_CHARGE
                                                        }, every_n_iter=1,
                                                       formatter=formatter_log)
        return


    def tfPrint(self, var, varStr='null', on=1):
        if on == 1:
            self.tmp = var
            return tf.Print(self.tmp, [self.tmp], message=varStr, summarize=40000)
        else:
            return self.tmp

    def _batch_scatter_nd_new(self, arg_sort_idx, sorted_value, N, output_shape):
        """
        batch级scatter_nd（原始scatter_nd不支持batch级）
        arg_sort_idx: 使用argsort后获得的下标, [batch_size,:]
        sorted_value: 使用sort排序后的值, [batch_size,:]
        N:            排序下标总长度，等于arg_sort_idx的长度
        output_shape: 输出维度[batch_size, X], X需要大于N, X大于N的列补0
        return:       使用arg_sort_idx,从sorted_value提取值,返回维度是output_shape的数据
        """
        y_sort = sorted_value
        ind = arg_sort_idx
        batch = tf.shape(y_sort)[0]
        N_matrix = tf.reshape(tf.tile(tf.reshape(tf.range(batch), [-1, 1]), [1, N]), [-1, N, 1])
        ind_matrix = tf.reshape(ind, [-1, N, 1])
        N_ind_matrix = tf.reshape(tf.concat([N_matrix, ind_matrix], axis=-1), [-1, 2])  # 构建坐标组合{[样本i, 排序j]}
        output = tf.scatter_nd(N_ind_matrix, tf.squeeze(tf.reshape(y_sort, [-1, 1]), axis=-1), output_shape)
        return output
