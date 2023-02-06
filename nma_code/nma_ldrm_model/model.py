# coding: utf-8 -*-
from data_input import *
from tools import *

import numpy as np
import tensorflow as tf


class DNN:
    def __init__(self):
        pass


    def _create_model(self, features):
        # output poi_feat = [BATCH_SIZE;LIST_NUM;POI_NUM;FEATURE_NUM]

        fc_out_mu = tf.concat([self.predict_feature], axis=-1)
        _shape = tf.shape(fc_out_mu)
        fc_out_lambda = tf.reshape(tf.concat([self.predict_feature], axis=-1), [-1, MAX_LIST_NUM, 2 * 2])
        fc_out_lambda = tf.concat([fc_out_lambda, self.set_encoder_feature], axis=-1)
        fc_out_lambda = self.tfPrint(fc_out_lambda, "fc_out_lambda")
        with tf.name_scope('rs_mu_lambda'):
            # mu与分配无关
            # rs_mu, rs_lambda = [BATCH_SIZE;LIST_NUM;POI_NUM;FEATURE_NUM]
            self.rs_mu = self._create_mu_network(fc_out_mu)
            self.rs_mu = tf.ones_like(self.rs_mu)
            self.rs_mu = self.tfPrint(self.rs_mu, "rs_mu")
            self.rs_lambda = self._create_lambda_network(fc_out_lambda)
            self.rs_lambda = self.tfPrint(self.rs_lambda, "rs_lambda")

        with tf.name_scope('rankscore_function'):
            # vvca_rs [-1, N], [-1, K]
            self.rs, self.top_list_idx = self._create_rankscore_function(self.list_poi_bid, self.list_poi_ctr, self.list_poi_gmv, self.rs_mu, rs_lambda=self.rs_lambda)
            self.rs = self.tfPrint(self.rs, "rs")

        with tf.name_scope('vvca_charge'):
            self.vvca_charge = self._create_vvca_payment(self.rs, self.top_list_idx, MAX_LIST_NUM)
            self.top_list_idx = self.tfPrint(self.top_list_idx, "self.top_list_idx")
            self.vvca_charge = self.tfPrint(self.vvca_charge, "self.vvca_charge")
            self.top_rs_mu = self.tfPrint(self.top_rs_mu, "self.top_rs_mu")
            self.output = tf.concat([tf.cast(self.top_list_idx, tf.float32), self.vvca_charge, self.top_rs_mu], axis=1)

        with tf.name_scope('vcg'):
            self.vcg_rs_mu = tf.ones_like(self.rs_mu)
            self.vcg_rs, self.vcg_top_list_idx = self._create_rankscore_function_base(self.list_poi_bid, self.list_poi_ctr,
                                                                                 self.list_poi_gmv, self.vcg_rs_mu)
            self.vcg_charge = self._create_vcg_payment(self.vcg_rs, self.vcg_top_list_idx, MAX_LIST_NUM)

        with tf.name_scope('uGSP'):
            self.ugsp_top_list_idx = self.vcg_top_list_idx
            self.ugsp_charge = tf.squeeze(tf.gather(self.list_poi_price, self.ugsp_top_list_idx, axis=1, batch_dims=-1),
                                          axis=1)

        with tf.name_scope('vvca_jfb'):
            VVCA_BID = tf.squeeze(tf.gather(self.list_poi_bid, self.top_list_idx, axis=1, batch_dims=-1), axis=1)
            self.jfb = tf.reduce_mean(self.vvca_charge) / tf.reduce_mean(VVCA_BID) # [Batch_size : 1]

    def _diff_module(self, features):
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
            fc_out = tf.nn.relu(fc_out)
        return tf.squeeze(fc_out, axis=-1)

    def _create_loss(self, labels):
        N = MAX_LIST_NUM
        # 主loss分位置加权
        Mr = self.Mr
        Mr = self.tfPrint(Mr, "Mr")
        _, reward_pay_tile = self._gen_reward(N, "pay*ctr")
        _, reward_gmv_tile = self._gen_reward(N, "gmv")
        _, reward_bid_tile = self._gen_reward(N, "bid*ctr")
        _, reward_ctr_tile = self._gen_reward(N, "ctr")
        _, reward_sw_tile = self._gen_reward(N, "sw")
        _, reward_rs_tile = self._gen_reward(N, "rs")
        all_reward, _ = self._gen_reward(N)
        self.loss_jfb = (tf.reduce_mean(self.jfb) - JFB_BASE) * (tf.reduce_mean(self.jfb) - JFB_BASE)
        reward_pay_tile = self.tfPrint(reward_pay_tile, "reward_pay_tile")
        self.loss_pay = -tf.reduce_mean(tf.reduce_sum(tf.multiply(Mr, reward_pay_tile), axis=[2, 1]))
        self.loss_pay_value = -tf.reduce_mean(tf.reduce_sum(reward_pay_tile, axis=[2, 1]))
        self.loss_pay = self.tfPrint(self.loss_pay, "loss_pay")
        self.loss_gmv = -tf.reduce_mean(tf.reduce_sum(tf.multiply(Mr, reward_gmv_tile), axis=[2, 1]))
        self.loss_gmv_value = -tf.reduce_mean(tf.reduce_sum(reward_gmv_tile, axis=[2, 1]))
        self.loss_bid = -tf.reduce_mean(tf.reduce_sum(tf.multiply(Mr, reward_bid_tile), axis=[2, 1]))
        self.loss_ctr = -tf.reduce_mean(tf.reduce_sum(tf.multiply(Mr, reward_ctr_tile), axis=[2, 1]))
        self.loss_sw = -tf.reduce_mean(tf.reduce_sum(tf.multiply(Mr, reward_sw_tile), axis=[2, 1]))
        self.loss_rs = -tf.reduce_mean(tf.reduce_sum(tf.multiply(Mr, reward_rs_tile), axis=[2, 1]))
        top_Mr = tf.squeeze(tf.gather(Mr, [0], axis=1, batch_dims=-1), axis=1)
        Mr_soft = tf.gather(top_Mr, self.vcg_top_list_idx, axis=1, batch_dims=-1)
        Mr_soft = self.tfPrint(Mr_soft, "self.Mr_soft")
        top_value = tf.math.top_k(top_Mr, k=1).values
        self.loss_ce = -tf.reduce_mean(Mr_soft)
        self.loss_ce = self.tfPrint(self.loss_ce, "self.loss_ce")
        self.loss_ce_value = tf.reduce_mean(top_value) - tf.reduce_mean(Mr_soft)
        self.loss = PAY_K * (self.loss_pay + GMVK * self.loss_gmv) + JFB_K * self.loss_jfb + SW_K * (self.loss_bid +  GMVK * self.loss_gmv) + CE_K * self.loss_ce + RS_K1 * self.loss_rs + \
                    CE_VALUE_K * self.loss_ce_value + PAY_VALUE_K * (self.loss_pay_value +  GMVK * self.loss_gmv)


    def _gen_reward(self, N, type="all"):
        if type == 'pay*ctr':
            # [BATCH_SIZE;LIST_NUM;LIST_SIZE]
            reward = tf.tile(tf.expand_dims(self.pay, axis=1), [1, N, 1]) * self.list_poi_ctr
        elif type == 'bid*ctr':
            reward = self.list_poi_bid * self.list_poi_ctr
        elif type == "ctr":
            reward = self.list_poi_ctr
        elif type == 'gmv':
            reward = self.list_poi_gmv
        elif type == 'sw':
            reward = self.list_poi_bid * self.list_poi_ctr + GMVK * self.list_poi_gmv
        elif type == 'rs':
            reward = tf.tile(tf.expand_dims(self.pay, axis=1), [1, N, 1]) * self.list_poi_ctr + GMVK * self.list_poi_gmv
        else:
            reward = self.list_poi_bid * self.list_poi_ctr
        # reward = self.tfPrint(reward, type)
        # reward_tile for Mr
        # self.top_list_idx : [BATCH_SIZE, K]
        reward = tf.reduce_sum(reward, axis=2) # [BATCH_SIZE;LIST_NUM]
        reward = self.tfPrint(reward, "reward")
        updates = tf.ones_like(self.top_list_idx)
        reward_mask = tf.cast(self._batch_scatter_nd_new(self.top_list_idx, updates, TOP_K, tf.shape(reward)), tf.float32)
        reward_mask = self.tfPrint(reward_mask, "reward_mask")
        reward = reward * reward_mask
        reward_tile = tf.reshape(tf.tile(reward, [1, N]), [-1, N, N])
        return reward, reward_tile

    def _create_rankscore_function(self, bid, ctr, gmv, rs_mu, rs_lambda=0, K=1):
        #output = bid * ctr * rs_mu + rs_lambda  # [bs, list_len, poi_len, 1]
        output = bid * ctr * rs_mu + rs_mu * GMVK * gmv   # [bs, list_len, poi_len]
        output = self.tfPrint(output, "output")
        output = tf.reduce_sum(output, axis=2) + rs_lambda # [bs, list_len]
        output = self.tfPrint(output, "output_sum")
        top_list_idx = tf.math.top_k(output, k=TOP_K).indices  # [-1, K]
        return output, top_list_idx

    def _create_rankscore_function_base(self, bid, ctr, gmv, rs_mu, rs_lambda=0, K=1):
        #output = bid * ctr * rs_mu + rs_lambda  # [bs, list_len, poi_len, 1]
        output = bid * ctr * rs_mu + rs_mu * GMVK_BASE * gmv   # [bs, list_len, poi_len]
        output = self.tfPrint(output, "output")
        output = tf.reduce_sum(output, axis=2) + rs_lambda # [bs, list_len]
        output = self.tfPrint(output, "output_sum")
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

    def _create_vvca_payment(self, rs, top_list_idx, N):
        # 生成mask矩阵
        # self.list_poi_id Batch_size:[-1, N, PER_LIST_POI_NUM]
        # top_list_idx Batch_size:[-1, 1, PER_LIST_POI_NUM]
        top_list_idx = self.tfPrint(top_list_idx, "top_list_idx")
        win_pois = tf.gather(self.list_poi_id, top_list_idx, axis=1, batch_dims=-1)
        tile_win_pois = tf.transpose(tf.tile(win_pois, [1, PER_LIST_POI_NUM, 1]), [0, 2, 1])
        tile_pois = tf.tile(tf.expand_dims(self.list_poi_id, axis=2), [1, 1, PER_LIST_POI_NUM, 1])  # -1, N, 3, 3
        tile_win_pois = self.tfPrint(tile_win_pois, "tile_win_pois")
        tile_pois = self.tfPrint(tile_pois, "tile_pois")
        mask = tf.cast(tf.transpose(tf.equal(tf.transpose(tile_pois, [1, 0, 2, 3]), tile_win_pois), [1, 0, 2, 3]), tf.int32) # [-1, N, 3, 3]
        mask = self.tfPrint(mask, "mask")
        self.mask = (tf.reduce_sum(mask, axis=3) - 1) * -1 # [-1, N, 3]

        self.mask = self.tfPrint(self.mask, "self.mask")
        win_score = tf.gather(rs, top_list_idx, axis=1, batch_dims=-1)
        win_score = self.tfPrint(win_score, "win_score")

        mask_score = tf.tile(tf.expand_dims(rs, axis=2), [1, 1, PER_LIST_POI_NUM]) * tf.cast(self.mask, tf.float32)
        mask_score_max = tf.reduce_max(mask_score, axis=1)
        mask_score_max = self.tfPrint(mask_score_max, "mask_score_max")

        self.rs_gap = tf.tile(win_score, [1, PER_LIST_POI_NUM]) - mask_score_max # [-1, 3]
        self.list_poi_bid = self.tfPrint(self.list_poi_bid, "self.list_poi_bid")
        list_poi_bid = tf.squeeze(tf.gather(self.list_poi_bid, top_list_idx, axis=1, batch_dims=-1), axis=1)
        list_poi_ctr = tf.squeeze(tf.gather(self.list_poi_ctr, top_list_idx, axis=1, batch_dims=-1), axis=1)
        self.top_rs_mu = tf.squeeze(tf.gather(self.rs_mu, top_list_idx, axis=1, batch_dims=-1), axis=1)

        list_poi_bid = self.tfPrint(list_poi_bid, "list_poi_bid")
        self.pay = list_poi_bid - self.rs_gap / (list_poi_ctr * self.top_rs_mu)  # [-1, 3]
        self.pay = tf.nn.relu(self.pay)
        return self.pay

    def _create_vcg_payment(self, rs, top_list_idx, N):
        # 生成mask矩阵
        # self.list_poi_id Batch_size:[-1, N, PER_LIST_POI_NUM]
        # top_list_idx Batch_size:[-1, 1, PER_LIST_POI_NUM]
        top_list_idx = self.tfPrint(top_list_idx, "top_list_idx")
        win_pois = tf.gather(self.list_poi_id, top_list_idx, axis=1, batch_dims=-1)
        tile_win_pois = tf.transpose(tf.tile(win_pois, [1, PER_LIST_POI_NUM, 1]), [0, 2, 1])
        tile_pois = tf.tile(tf.expand_dims(self.list_poi_id, axis=2), [1, 1, PER_LIST_POI_NUM, 1])  # -1, N, 3, 3
        tile_win_pois = self.tfPrint(tile_win_pois, "tile_win_pois")
        tile_pois = self.tfPrint(tile_pois, "tile_pois")
        mask = tf.cast(tf.transpose(tf.equal(tf.transpose(tile_pois, [1, 0, 2, 3]), tile_win_pois), [1, 0, 2, 3]),
                       tf.int32)  # [-1, N, 3, 3]
        mask = self.tfPrint(mask, "mask")
        self.mask = (tf.reduce_sum(mask, axis=3) - 1) * -1  # [-1, N, 3]

        self.mask = self.tfPrint(self.mask, "self.mask")
        win_score = tf.gather(rs, top_list_idx, axis=1, batch_dims=-1)
        win_score = self.tfPrint(win_score, "win_score")

        mask_score = tf.tile(tf.expand_dims(rs, axis=2), [1, 1, PER_LIST_POI_NUM]) * tf.cast(self.mask, tf.float32)
        mask_score_max = tf.reduce_max(mask_score, axis=1)
        mask_score_max = self.tfPrint(mask_score_max, "mask_score_max")

        self.rs_gap = tf.tile(win_score, [1, PER_LIST_POI_NUM]) - mask_score_max  # [-1, 3]
        list_poi_bid = tf.squeeze(tf.gather(self.list_poi_bid, top_list_idx, axis=1, batch_dims=-1), axis=1)
        list_poi_ctr = tf.squeeze(tf.gather(self.list_poi_ctr, top_list_idx, axis=1, batch_dims=-1), axis=1)
        self.top_vcg_rs_mu = tf.squeeze(tf.gather(self.vcg_rs_mu, top_list_idx, axis=1, batch_dims=-1), axis=1)

        list_poi_bid = self.tfPrint(list_poi_bid, "list_poi_bid")

        pay = list_poi_bid - self.rs_gap / (list_poi_ctr * self.top_vcg_rs_mu)  # [-1, 3]
        return pay

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())


    def _point_to_list_features(self, features):
        # 'poi_dense_feature': poi_dense_feature,
        # 'poi_sub_feature': poi_sub_feature,
        # 'poi_size': poi_size
        # position + adid + objecttype + ctr + locationid + categoryid + price + iscontext
        # features['poi_sub_feature']
        position = tf.squeeze(tf.gather(features['poi_dense_feature'], list(range(0, 1)), axis=-1), axis=-1)
        adid = tf.squeeze(tf.gather(features['poi_dense_feature'], list(range(1, 2)), axis=-1), axis=-1)
        objecttype = tf.squeeze(tf.gather(features['poi_dense_feature'], list(range(2, 3)), axis=-1), axis=-1)
        ctr = tf.squeeze(tf.gather(features['poi_dense_feature'], list(range(3, 4)), axis=-1), axis=-1)
        locationid = tf.squeeze(tf.gather(features['poi_dense_feature'], list(range(4, 5)), axis=-1), axis=-1)
        categoryid = tf.squeeze(tf.gather(features['poi_dense_feature'], list(range(5, 6)), axis=-1), axis=-1)
        price = tf.squeeze(tf.gather(features['poi_dense_feature'], list(range(6, 7)), axis=-1), axis=-1)
        iscontext = tf.squeeze(tf.gather(features['poi_dense_feature'], list(range(7, 8)), axis=-1), axis=-1)
        poi_size = features['poi_size']
        bid = features['bid']

        gather_idx = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                      [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
                      [2, 0], [2, 1], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
                      [3, 0], [3, 1], [3, 2], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
                      [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9],
                      [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 6], [5, 7], [5, 8], [5, 9],
                      [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 7], [6, 8], [6, 9],
                      [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 8], [7, 9],
                      [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 9],
                      [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8]]

        self.list_poi_id = tf.gather(adid, gather_idx, axis=-1)
        self.list_poi_ctr = tf.gather(ctr, gather_idx, axis=-1)
        min_ctr = tf.ones_like(self.list_poi_ctr) * 0.001
        self.list_poi_ctr = tf.where(self.list_poi_ctr > 0.0, self.list_poi_ctr, min_ctr)
        self.list_poi_ctr = tf.where(self.list_poi_ctr < 0.1, self.list_poi_ctr, min_ctr)
        self.list_poi_bid = tf.gather(bid, gather_idx, axis=-1)

        self.list_poi_cpm = self.list_poi_ctr * self.list_poi_bid
        self.list_poi_rk = self.list_poi_cpm
        self.list_poi_gmv = self.list_poi_ctr

        self.list_poi_price = self.list_poi_bid * 0.8

        self.predict_feature = tf.stack([self.list_poi_ctr, self.list_poi_cpm], axis=-1)



        # set encoder特征
        # ctr, bid, cpm 均值， 最大值
        # 与列表的相对差值 绝对值差值 最大值差值
        mean_list_ctr = tf.reduce_mean(self.list_poi_ctr, axis=2)
        mean_ctr = tf.tile(tf.reshape(tf.reduce_mean(mean_list_ctr, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_ctr = tf.tile(tf.reshape(tf.reduce_max(mean_list_ctr, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        ctr_f_1 = mean_list_ctr - mean_ctr
        ctr_f_2 = (mean_list_ctr - mean_ctr) / mean_ctr
        ctr_f_3 = mean_list_ctr - max_ctr
        ctr_f_4 = (mean_list_ctr - max_ctr) / max_ctr

        mean_list_bid = tf.reduce_mean(self.list_poi_bid, axis=2)
        mean_bid = tf.tile(tf.reshape(tf.reduce_mean(mean_list_bid, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_bid = tf.tile(tf.reshape(tf.reduce_max(mean_list_bid, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        bid_f_1 = mean_list_bid - mean_bid
        bid_f_2 = (mean_list_bid - mean_bid) / mean_bid
        bid_f_3 = mean_list_bid - max_bid
        bid_f_4 = (mean_list_bid - max_bid) / max_bid

        mean_list_cpm = tf.reduce_mean(self.list_poi_cpm, axis=2)
        mean_cpm = tf.tile(tf.reshape(tf.reduce_mean(mean_list_cpm, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_cpm = tf.tile(tf.reshape(tf.reduce_max(mean_list_cpm, axis=1),  [-1, 1]), [1, MAX_LIST_NUM])
        cpm_f_1 = mean_list_cpm - mean_cpm
        cpm_f_2 = (mean_list_cpm - mean_cpm) / mean_cpm
        cpm_f_3 = mean_list_cpm - max_cpm
        cpm_f_4 = (mean_list_cpm - max_cpm) / max_cpm

        mean_list_rk = tf.reduce_mean(self.list_poi_rk, axis=2)
        top_2_list_value = tf.math.top_k(mean_list_rk, k=2).values
        mean_rk = tf.tile(tf.reshape(tf.reduce_mean(mean_list_rk, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_rk = tf.tile(tf.reshape(tf.reduce_max(mean_list_rk, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        top_2_rk = tf.tile(tf.reshape(tf.reduce_min(top_2_list_value, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        rk_f_1 = mean_list_rk - mean_rk
        rk_f_2 = (mean_list_rk - mean_rk) / mean_rk
        rk_f_3 = mean_list_rk - max_rk
        rk_f_4 = (mean_list_rk - max_rk) / max_rk
        rk_f_5 = mean_list_rk - top_2_rk
        rk_f_6 = (mean_list_rk - top_2_rk) / top_2_rk

        self.set_encoder_feature = tf.stack([ctr_f_1, ctr_f_2, ctr_f_3, ctr_f_4,
                                         bid_f_1, bid_f_2, bid_f_3, bid_f_4,
                                         cpm_f_1, cpm_f_2, cpm_f_3, cpm_f_4,
                                         rk_f_1, rk_f_2, rk_f_3, rk_f_4, rk_f_5, rk_f_6], axis=-1)

        return features

    def _process_features(self, features):
        # N * M * K
        # N * D ( D <= M )
        self.list_poi_predict_feature = tf.reshape(features['screen_predict_feature'],
                                                   [-1, MAX_LIST_NUM, PER_LIST_POI_NUM, POI_PREDICT_FEA_NUM])
        self.screen_dense_feature = tf.reshape(features['screen_dense_feature'],
                                                   [-1, MAX_LIST_NUM, PER_LIST_POI_NUM, POI_DENSE_FEA_NUM])
        self.screen_cate_feature = tf.reshape(features['screen_cate_feature'],
                                               [-1, MAX_LIST_NUM, PER_LIST_POI_NUM, POI_CATE_FEA_NUM])


        _shape = tf.shape(self.screen_dense_feature)
        mean, sted = get_normalization_parameter(MEAN_VAR_PATH_POI)
        mean = tf.constant(name='mean_cc', shape=(1, POI_DENSE_FEA_NUM),
                                value=mean,
                                dtype=tf.float32)
        sted = tf.constant(name='sted_cc', shape=(1, POI_DENSE_FEA_NUM),
                                value=sted,
                                dtype=tf.float32)
        mean = tf.tile(tf.reshape(mean, [1, 1, 1, POI_DENSE_FEA_NUM]), [_shape[0], _shape[1], _shape[2], 1])
        sted = tf.tile(tf.reshape(sted, [1, 1, 1, POI_DENSE_FEA_NUM]), [_shape[0], _shape[1], _shape[2], 1])
        self.screen_dense_feature = (self.screen_dense_feature - mean) / sted
        
        self.predict_feature = tf.gather(self.list_poi_predict_feature, list(range(1, 5)), axis=3)
        self.predict_feature = self.tfPrint(self.predict_feature, "predict_feature")
        self.list_poi_id = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(0, 1)), axis=-1), axis=-1)
        self.list_poi_id = self.tfPrint(self.list_poi_id, "self.list_poi_id")
        self.list_poi_ctr = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(1, 2)), axis=-1), axis=-1)
        self.list_poi_imp = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(2, 3)), axis=-1), axis=-1)
        self.list_poi_ctr = self.list_poi_ctr * self.list_poi_imp
        min_ctr = tf.ones_like(self.list_poi_ctr) * 0.001
        self.list_poi_ctr = tf.where(self.list_poi_ctr > 0, self.list_poi_ctr, min_ctr)
        self.list_poi_cvr = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(3, 4)), axis=-1), axis=-1)
        min_cvr = tf.ones_like(self.list_poi_cvr) * 0.01
        self.list_poi_cvr = tf.where(self.list_poi_cvr > 0, self.list_poi_cvr, min_cvr)
        self.list_poi_pprice = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(4, 5)), axis=-1), axis=-1)
        self.list_poi_bid = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(5, 6)), axis=-1), axis=-1)
        min_bid = tf.ones_like(self.list_poi_bid) * 10.0
        self.list_poi_bid = tf.where(self.list_poi_bid > 0, self.list_poi_bid, min_bid)
        self.list_poi_gspprice = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(6, 7)), axis=-1), axis=-1)
        self.list_poi_price = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(7, 8)), axis=-1), axis=-1)
        self.list_poi_gmv = self.list_poi_pprice * self.list_poi_ctr * self.list_poi_cvr
        min_gmv = tf.ones_like(self.list_poi_gmv) * 0.1
        self.list_poi_gmv = tf.where(self.list_poi_gmv > 0, self.list_poi_gmv, min_gmv)
        self.list_poi_cpm = self.list_poi_ctr * self.list_poi_bid
        self.list_poi_ctr = self.tfPrint(self.list_poi_ctr, "self.list_poi_ctr")
        self.list_poi_bid = self.tfPrint(self.list_poi_bid, "self.list_poi_bid")
        self.list_poi_cpm = self.tfPrint(self.list_poi_cpm, "self.list_poi_cpm")
        self.list_poi_gmv = self.tfPrint(self.list_poi_gmv, "self.list_poi_gmv")
        self.list_poi_rk = self.list_poi_cpm + GMVK * self.list_poi_gmv


        # set encoder特征
        # ctr, bid, cpm 均值， 最大值
        # 与列表的相对差值 绝对值差值 最大值差值
        mean_list_ctr = tf.reduce_mean(self.list_poi_ctr, axis=2)
        mean_ctr = tf.tile(tf.reshape(tf.reduce_mean(mean_list_ctr, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_ctr = tf.tile(tf.reshape(tf.reduce_max(mean_list_ctr, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        ctr_f_1 = mean_list_ctr - mean_ctr
        ctr_f_2 = (mean_list_ctr - mean_ctr) / mean_ctr
        ctr_f_3 = mean_list_ctr - max_ctr
        ctr_f_4 = (mean_list_ctr - max_ctr) / max_ctr

        mean_list_bid = tf.reduce_mean(self.list_poi_bid, axis=2)
        mean_bid = tf.tile(tf.reshape(tf.reduce_mean(mean_list_bid, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_bid = tf.tile(tf.reshape(tf.reduce_max(mean_list_bid, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        bid_f_1 = mean_list_bid - mean_bid
        bid_f_2 = (mean_list_bid - mean_bid) / mean_bid
        bid_f_3 = mean_list_bid - max_bid
        bid_f_4 = (mean_list_bid - max_bid) / max_bid

        mean_list_cpm = tf.reduce_mean(self.list_poi_cpm, axis=2)
        mean_cpm = tf.tile(tf.reshape(tf.reduce_mean(mean_list_cpm, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_cpm = tf.tile(tf.reshape(tf.reduce_max(mean_list_cpm, axis=1),  [-1, 1]), [1, MAX_LIST_NUM])
        cpm_f_1 = mean_list_cpm - mean_cpm
        cpm_f_2 = (mean_list_cpm - mean_cpm) / mean_cpm
        cpm_f_3 = mean_list_cpm - max_cpm
        cpm_f_4 = (mean_list_cpm - max_cpm) / max_cpm

        mean_list_gmv = tf.reduce_mean(self.list_poi_gmv, axis=2)
        mean_gmv = tf.tile(tf.reshape(tf.reduce_mean(mean_list_gmv, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_gmv = tf.tile(tf.reshape(tf.reduce_max(mean_list_gmv, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        gmv_f_1 = mean_list_gmv - mean_gmv
        gmv_f_2 = (mean_list_gmv - mean_gmv) / mean_gmv
        gmv_f_3 = mean_list_gmv - max_gmv
        gmv_f_4 = (mean_list_gmv - max_gmv) / max_gmv

        mean_list_rk = tf.reduce_mean(self.list_poi_rk, axis=2)
        top_2_list_value = tf.math.top_k(mean_list_rk, k=2).values
        mean_rk = tf.tile(tf.reshape(tf.reduce_mean(mean_list_rk, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        max_rk = tf.tile(tf.reshape(tf.reduce_max(mean_list_rk, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        top_2_rk = tf.tile(tf.reshape(tf.reduce_min(top_2_list_value, axis=1), [-1, 1]), [1, MAX_LIST_NUM])
        rk_f_1 = mean_list_rk - mean_rk
        rk_f_2 = (mean_list_rk - mean_rk) / mean_rk
        rk_f_3 = mean_list_rk - max_rk
        rk_f_4 = (mean_list_rk - max_rk) / max_rk
        rk_f_5 = mean_list_rk - top_2_rk
        rk_f_6 = (mean_list_rk - top_2_rk) / top_2_rk

        self.set_encoder_feature = tf.stack([ctr_f_1, ctr_f_2, ctr_f_3, ctr_f_4,
                                              bid_f_1, bid_f_2, bid_f_3, bid_f_4,
                                              cpm_f_1, cpm_f_2, cpm_f_3, cpm_f_4,
                                              rk_f_1, rk_f_2, rk_f_3, rk_f_4, rk_f_5, rk_f_6,
                                            gmv_f_1, gmv_f_2, gmv_f_3, gmv_f_4], axis=-1)
        return features

    def model_fn_estimator(self, features, labels, mode, params):
        # self._process_features(features)
        self._point_to_list_features(features)
        self._create_model(features)

        if mode == tf.estimator.ModeKeys.TRAIN:
            self._diff_module(features)
            self._create_loss(labels)
            self._create_optimizer()
            self._build_indicator(labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op,
                                              training_hooks=[self.logging_hook])
        else:
            if 'save_model' in list(params.keys()):
                outputs = {
                    "output": tf.identity(self.output, "output")
                    }
            else:
                VVCA_CTR = tf.squeeze(tf.gather(self.list_poi_ctr, self.top_list_idx, axis=1, batch_dims=-1), axis=1)
                VVCA_BID = tf.squeeze(tf.gather(self.list_poi_bid, self.top_list_idx, axis=1, batch_dims=-1), axis=1)
                VVCA_GMV = tf.squeeze(tf.gather(self.list_poi_gmv, self.top_list_idx, axis=1, batch_dims=-1), axis=1)
                VVCA_CHARGE = self.vvca_charge
                VVCA_CPM = self.vvca_charge * VVCA_CTR
                VVCA_JFB = VVCA_CHARGE / VVCA_BID
                VVCA_SW = VVCA_CTR * VVCA_BID

                VCG_CTR = tf.squeeze(tf.gather(self.list_poi_ctr, self.vcg_top_list_idx, axis=1, batch_dims=-1), axis=1)
                VCG_BID = tf.squeeze(tf.gather(self.list_poi_bid, self.vcg_top_list_idx, axis=1, batch_dims=-1), axis=1)
                VCG_GMV = tf.squeeze(tf.gather(self.list_poi_gmv, self.vcg_top_list_idx, axis=1, batch_dims=-1), axis=1)
                VCG_CHARGE = self.vcg_charge
                VCG_CPM = self.vcg_charge * VCG_CTR
                VCG_JFB = VCG_CHARGE / VCG_BID
                VCG_SW = VCG_CTR * VCG_BID

                UGSP_CTR = tf.squeeze(tf.gather(self.list_poi_ctr, self.ugsp_top_list_idx, axis=1, batch_dims=-1),
                                      axis=1)
                UGSP_BID = tf.squeeze(tf.gather(self.list_poi_bid, self.ugsp_top_list_idx, axis=1, batch_dims=-1),
                                      axis=1)
                UGSP_GMV = tf.squeeze(tf.gather(self.list_poi_gmv, self.ugsp_top_list_idx, axis=1, batch_dims=-1), axis=1)
                UGSP_CHARGE = self.ugsp_charge
                UGSP_CPM = self.ugsp_charge * UGSP_CTR
                UGSP_JFB = UGSP_CHARGE / UGSP_BID
                UGSP_SW = UGSP_CTR * UGSP_BID

                outputs = {'output': self.output,
                           'VVCA_CTR': VVCA_CTR,
                           'VVCA_CPM': VVCA_CPM,
                           'VVCA_BID': VVCA_BID,
                           'VVCA_CHARGE': VVCA_CHARGE,
                           'VVCA_SW': VVCA_SW,
                           'VVCA_GMV': VVCA_GMV,
                           'VCG_CTR': VCG_CTR,
                           'VCG_CPM': VCG_CPM,
                           'VCG_BID': VCG_BID,
                           'VCG_CHARGE': VCG_CHARGE,
                           'VCG_SW': VCG_SW,
                           'VCG_GMV': VCG_GMV,
                           'UGSP_CTR': UGSP_CTR,
                           'UGSP_CPM': UGSP_CPM,
                           'UGSP_BID': UGSP_BID,
                           'UGSP_CHARGE': UGSP_CHARGE,
                           'UGSP_SW': UGSP_SW,
                           'UGSP_GMV': UGSP_GMV
                           }
            export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                      tf.estimator.export.PredictOutput(outputs)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)

    def _build_indicator(self, labels):
        # GMV, BID, uGSP_CHARGE, VVCA_CHARGE, JFB
        # labels_result['charge']， self.list_poi_bid, self.list_poi_ctr, self.list_poi_cvr, self.list_poi_pprice

        VVCA_CTR = tf.squeeze(tf.gather(self.list_poi_ctr, self.top_list_idx, axis=1, batch_dims=-1), axis=1)
        VVCA_BID = tf.squeeze(tf.gather(self.list_poi_bid, self.top_list_idx, axis=1, batch_dims=-1), axis=1)
        VVCA_GMV = tf.squeeze(tf.gather(self.list_poi_gmv, self.top_list_idx, axis=1, batch_dims=-1), axis=1)
        VVCA_CHARGE = self.vvca_charge
        VVCA_CPM = self.vvca_charge * VVCA_CTR
        VVCA_JFB = tf.reduce_mean(VVCA_CHARGE) / tf.reduce_mean(VVCA_BID)
       
        VCG_CTR = tf.squeeze(tf.gather(self.list_poi_ctr, self.vcg_top_list_idx, axis=1, batch_dims=-1), axis=1)
        VCG_BID = tf.squeeze(tf.gather(self.list_poi_bid, self.vcg_top_list_idx, axis=1, batch_dims=-1), axis=1)
        VCG_GMV = tf.squeeze(tf.gather(self.list_poi_gmv, self.vcg_top_list_idx, axis=1, batch_dims=-1), axis=1)
        VCG_CHARGE = self.vcg_charge
        VCG_CPM = self.vcg_charge * VCG_CTR 
        VCG_JFB = tf.reduce_mean(VCG_CHARGE) / tf.reduce_mean(VCG_BID)

        UGSP_CTR = tf.squeeze(tf.gather(self.list_poi_ctr, self.ugsp_top_list_idx, axis=1, batch_dims=-1), axis=1)
        UGSP_BID = tf.squeeze(tf.gather(self.list_poi_bid, self.ugsp_top_list_idx, axis=1, batch_dims=-1), axis=1)
        UGSP_GMV = tf.squeeze(tf.gather(self.list_poi_gmv, self.ugsp_top_list_idx, axis=1, batch_dims=-1), axis=1)
        UGSP_CHARGE = self.ugsp_charge
        UGSP_CPM = self.ugsp_charge * UGSP_CTR 
        UGSP_JFB = tf.reduce_mean(UGSP_CHARGE) / tf.reduce_mean(UGSP_BID)


        def formatter_log(tensors):
            log_string = "indicator_info: step {}:, loss={:.4f}, loss_pay={:.4f}, loss_jfb={:.4f}" \
                          "VVCA_CHARGE={:.4f}, VVCA_CPM={:.4f}, VVCA_JFB={:.4f}," \
                          "UGSP_CHARGE={:.4f}, UGSP_CPM={:.4f}, UGSP_JFB={:.4f}," \
                          "VCG_CHARGE={:.4f}, VCG_CPM={:.4f}, VCG_JFB={:.4f}," \
                          "VVCA_CTR={:.4f}, VCG_CTR={:.4f}, UGSP_CTR={:.4f}" \
                          "VVCA_GMV={:.4f}, VCG_GMV={:.4f}, UGSP_GMV={:.4f}" \
                          "VVCA_VCG_GMV_IMPROVE={:.4f}, VVCA_GSP_GMV_IMPROVE={:.4f}" \
                         "VVCA_VCG_CPM_IMPROVE={:.4f}, VVCA_GSP_CPM_IMPROVE={:.4f}, VVCA_VCG_JFB_IMPROVE={:.4f}, VVCA_GSP_JFB_IMPROVE={:.4f},".format(
                tensors["step"], tensors["loss"], tensors["loss_pay"], tensors["loss_jfb"], tensors["VVCA_CHARGE"], tensors["VVCA_CPM"], tensors["VVCA_JFB"],
                tensors["UGSP_CHARGE"], tensors["UGSP_CPM"], tensors["UGSP_JFB"],
                tensors["VCG_CHARGE"], tensors["VCG_CPM"], tensors["VCG_JFB"],
                tensors["VVCA_CTR"], tensors["VCG_CTR"], tensors["UGSP_CTR"],
                tensors["VVCA_GMV"], tensors["VCG_GMV"], tensors["UGSP_GMV"],
                tensors["VVCA_VCG_GMV_IMPROVE"], tensors["VVCA_GSP_GMV_IMPROVE"],
                tensors["VVCA_VCG_CPM_IMPROVE"], tensors["VVCA_GSP_CPM_IMPROVE"], tensors["VVCA_VCG_JFB_IMPROVE"], tensors["VVCA_GSP_JFB_IMPROVE"],
            )
            return "\n" + log_string

        self.logging_hook = tf.train.LoggingTensorHook({"loss": self.loss,
                                                        "loss_pay": self.loss_pay,
                                                        "loss_jfb": self.loss_jfb,
                                                        "step": tf.train.get_global_step(),
                                                        "VVCA_CHARGE": tf.reduce_mean(VVCA_CHARGE),
                                                        "VVCA_CPM": tf.reduce_mean(VVCA_CPM),
                                                        "VVCA_JFB": tf.reduce_mean(VVCA_JFB),
                                                        
                                                        "UGSP_CHARGE": tf.reduce_mean(UGSP_CHARGE),
                                                        "UGSP_CPM": tf.reduce_mean(UGSP_CPM),
                                                        "UGSP_JFB": tf.reduce_mean(UGSP_JFB),
                                                        
                                                        "VCG_CHARGE": tf.reduce_mean(VCG_CHARGE),
                                                        "VCG_CPM": tf.reduce_mean(VCG_CPM),
                                                        "VCG_JFB": tf.reduce_mean(VCG_JFB),

                                                        "VVCA_CTR": tf.reduce_mean(VVCA_CTR),
                                                        "VCG_CTR": tf.reduce_mean(VCG_CTR),
                                                        "UGSP_CTR": tf.reduce_mean(UGSP_CTR),

                                                        "VVCA_GMV": tf.reduce_mean(VVCA_GMV),
                                                        "VCG_GMV": tf.reduce_mean(VCG_GMV),
                                                        "UGSP_GMV": tf.reduce_mean(UGSP_GMV),

                                                        "VVCA_VCG_CPM_IMPROVE": (tf.reduce_mean(VVCA_CPM) - tf.reduce_mean(VCG_CPM)) / tf.reduce_mean(VCG_CPM),
                                                        "VVCA_GSP_CPM_IMPROVE": (tf.reduce_mean(VVCA_CPM) - tf.reduce_mean(UGSP_CPM)) / tf.reduce_mean(UGSP_CPM),
                                                        "VVCA_VCG_GMV_IMPROVE": (tf.reduce_mean(VVCA_GMV) - tf.reduce_mean(VCG_GMV)) / tf.reduce_mean(VCG_GMV),
                                                        "VVCA_GSP_GMV_IMPROVE": (tf.reduce_mean(VVCA_GMV) - tf.reduce_mean(UGSP_GMV)) / tf.reduce_mean(UGSP_GMV),
                                                        "VVCA_VCG_JFB_IMPROVE": (tf.reduce_mean(VVCA_JFB) - tf.reduce_mean(VCG_JFB)),
                                                        "VVCA_GSP_JFB_IMPROVE": (tf.reduce_mean(VVCA_JFB) - tf.reduce_mean(UGSP_JFB)),
                                                        }, every_n_iter=1000,
                                                       formatter=formatter_log)
        return


    def tfPrint(self, var, varStr='null', on=0):
        if on == 1:
            self.tmp = var
            return tf.Print(self.tmp, [self.tmp], message=varStr, summarize=100)
        else:
            self.tmp = var
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
