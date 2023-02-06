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
        self.tar = 1.0
        with tf.name_scope('mean_sted'):
            mean, sted = get_normalization_parameter(MEAN_VAR_PATH)
            bias_weight = get_bias_weight_parameter(BIAS_WEIGHT_PATH)
            self.mean = tf.constant(name='mean', shape=(1, POI_DENSE_FEA_NUM),
                                    value=mean,
                                    dtype=tf.float32)
            self.sted = tf.constant(name='sted', shape=(1, POI_DENSE_FEA_NUM),
                                    value=sted,
                                    dtype=tf.float32)
            self.bias_weight = tf.constant(name='bias_weight', shape=(1, MAX_LIST_NUM),
                                           value=bias_weight,
                                           dtype=tf.float32)

        with tf.name_scope('feature_emb_weights'):
            # 初始化为均值为0，标准差为0.01的正态分布
            if USE_PRE_TRAINED_LIST_WISE_MODEL:
                self.feature_emb_weights = \
                    {'cate_feature_embeddings': tf.get_variable('cate_feature_embeddings',
                                                                shape=[CATE_VOCAB_SIZE, LWP['CATE_EMBEDDING_SIZE']],
                                                                initializer=tf.random_normal_initializer(0.0, 0.01,
                                                                                                         seed=RANDOM_SEED),
                                                                trainable=False)
                     }
            else:
                self.feature_emb_weights = \
                    {'cate_feature_embeddings': tf.get_variable('cate_feature_embeddings',
                                                                shape=[CATE_VOCAB_SIZE, CATE_EMBEDDING_SIZE],
                                                                initializer=tf.random_normal_initializer(0.0, 0.01,
                                                                                                         seed=RANDOM_SEED),
                                                                trainable=True)
                     }

        with tf.name_scope('mu_network_weights'):
            self.mu_network_weights = self._create_network_weights(MU_MLP, INPUT_DIM, "mu_net")
        with tf.name_scope('lambda_network_weights'):
            self.lambda_network_weights = self._create_network_weights(LAMBDA_MLP, INPUT_DIM, "lambda_net")

    def _create_network_weights(self, network_deep_layers, network_input_dim, weight_prefix, trainable=True):
        network_weights = {}
        network_num_layer = len(network_deep_layers)
        for i in range(0, network_num_layer):
            if i == 0:
                last_layer_size = network_input_dim
            else:
                last_layer_size = network_deep_layers[i - 1]

            glorot = np.sqrt(2.0 / (last_layer_size + network_deep_layers[i]))

            network_weights['layer_%d' % i] = tf.get_variable(name='%s_layer_%d' % (weight_prefix, i),
                                                              shape=[last_layer_size, network_deep_layers[i]],
                                                              initializer=tf.random_normal_initializer(0.0,
                                                                                                       glorot,
                                                                                                       seed=RANDOM_SEED),
                                                              dtype=np.float32,
                                                              trainable=trainable)
            network_weights['bias_%d' % i] = tf.get_variable(name='%s_bias_%d' % (weight_prefix, i),
                                                             shape=[1, network_deep_layers[i]],
                                                             initializer=tf.zeros_initializer,
                                                             dtype=np.float32,
                                                             trainable=trainable)
        return network_weights

    def _init_feature(self, features):
        # reshape input
        self.list_poi_predict_feature = features['list_poi_predict_feature']
        self.list_poi_dense_feature = features['list_poi_dense_feature']
        self.list_poi_mask_feature = features['list_poi_mask_feature']
        self.list_poi_cate_idx = features['list_poi_cate_feature']
        self.list_size = tf.minimum(features['list_size'], MAX_LIST_NUM)
        self.list_poi_cate_embedding_lookupfeature = tf.reshape(
            tf.nn.(self.feature_emb_weights['cate_feature_embeddings'], self.list_poi_cate_idx),
            [-1, MAX_LIST_NUM, PER_LIST_POI_NUM, POI_CATE_FEA_NUM * CATE_EMBEDDING_SIZE])
        # split feature   [bs, list_size, poi_size]
        self.list_poi_bid = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(0, 1)), axis=-1), axis=-1)
        self.list_poi_ctr = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(1, 2)), axis=-1), axis=-1)
        self.list_poi_cvr = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(2, 3)), axis=-1), axis=-1)
        self.list_poi_gmv = tf.squeeze(tf.gather(self.list_poi_predict_feature, list(range(3, 4)), axis=-1), axis=-1)
        zero_tmp = tf.zeros_like(self.list_poi_ctr)
        self.list_poi_pprice = tf.where(tf.greater(self.list_poi_ctr * self.list_poi_cvr, zero_tmp),
                                        tf.div(self.list_poi_gmv, self.list_poi_ctr * self.list_poi_cvr), zero_tmp)
        # normalization
        mean = tf.tile(tf.expand_dims(self.mean, axis=0), [tf.shape(self.list_poi_dense_feature)[0], MAX_LIST_NUM, 1])
        sted = tf.tile(tf.expand_dims(self.sted, axis=0), [tf.shape(self.list_poi_dense_feature)[0], MAX_LIST_NUM, 1])
        self.list_poi_dense_feature = (self.list_poi_dense_feature - mean) / sted
        # build mask & avg_feature
        self.list_len_mask = tf.squeeze(tf.sequence_mask(lengths=self.list_size, maxlen=MAX_LIST_NUM, dtype=tf.float32), axis=1)
        self.list_poi_feature = tf.concat([self.list_poi_dense_feature, self.list_poi_cate_feature], axis=-1)
        # feature mask
        if self.feature_mask_on:
            self.list_poi_feature = tf.gather(self.list_poi_feature, self.feature_mask, axis=-1)

    def _list_wise_predict(self, features):
        # read pre-trained list_wise model check_point
        if USE_PRE_TRAINED_LIST_WISE_MODEL:
            # gen list-wise model used features
            self._init_list_wise_model_feature(features)
            # create p,a network weights, embedding weights
            self._create_list_wise_model_weights()
            # init weights from ck
            self._init_list_wise_weights_from_checkpoint()
            # predict list-wise ctr  [bs, list_size, poi_size]
            self.list_poi_ctr = tf.reshape(self._create_list_wise_model(), [-1, MAX_LIST_NUM, PER_LIST_POI_NUM])
            self.list_poi_gmv = self.list_poi_ctr * self.list_poi_cvr * self.list_poi_pprice

    def _create_rankscore_function(self, bid, ctr, gmv, rs_mu, rs_lambda, K):
        output = bid * ctr * rs_mu + rs_lambda  # [bs, list_len, poi_len, 1]
        output = tf.reduce_sum(output, axis=2)  # [bs, list_len, 1]
        # todo topk的idx，目前仅支持top1
        top1_list_idx = tf.math.top_k(output, k=K).indices  # [-1, K]
        return output, top1_list_idx

    def _create_differentiable_sorting(self, rs, top_list_idx, list_set_size, N, K):
        list_mask = tf.reduce_sum(tf.one_hot(top_list_idx, depth=N, axis=-1), axis=-2)  # [-1, N]
        list_mask_matrix = tf.tile(tf.expand_dims(list_mask, axis=-1), [1, 1, N])  # [-1, N]
        Ar1 = tf.reshape(tf.tile(rs, [1, N]), [-1, N, N])
        Ar2 = tf.transpose(Ar1, perm=[0, 2, 1])
        Ar = tf.abs(Ar1 - Ar2)  # [-1, N, N]
        ArOne = tf.reduce_sum(Ar, axis=2)
        ArOne_matrix = tf.tile(tf.reshape(ArOne, [-1, 1, N]), [1, N, 1])  # [-1, N, N]
        input_matrix = tf.tile(tf.reshape(rs, [-1, 1, N]), [1, N, 1])  # [-1, N, N]
        k = tf.cast(tf.range(start=1, limit=N + 1), dtype=tf.float32)
        W = tf.tile(tf.reshape(list_set_size + 1 - 2 * k, [-1, N, 1]), [1, 1, N])
        Ck = tf.multiply(input_matrix, tf.cast(W, dtype=tf.float32)) - ArOne_matrix
        # Ck = self.tfPrint(Ck,'Ck1')
        Ck = tf.where(tf.equal(list_mask_matrix, 1), Ck, tf.ones_like(Ck) * (-1e8))
        # Ck = self.tfPrint(Ck,'Ck2')
        Mr = tf.nn.softmax(Ck / self.tar, axis=2)
        # Mr = self.tfPrint(Mr, 'Mr')
        Mr = tf.gather(Mr, list(range(PAY_LIST_NUM)), axis=1)  # 提出Mr的top列
        return Mr



    def _create_model(self, features):
        self._init_feature(features)
        with tf.name_scope('list_wise_predict'):
            self._list_wise_predict(features)
        with tf.name_scope('rs_mu'):
            # poi_wise mu  [-1, N, 4]
            # mu与分配无关，相同POI的mu值一样
            self.rs_mu = self._create_mu_network(self.list_poi_feature)
        with tf.name_scope('rs_lambda'):
            # list_wise lambda [-1, N]
            # lambda可以与分配有关，使用预估CTR
            self.rs_lambda = self._create_lambda_network(self.list_poi_feature)
        with tf.name_scope('rankscore_function'):
            # vvca_rs [-1, N], [-1, K]
            self.rs, self.top_list_idx = self._create_rankscore_function(self.list_poi_bid, self.list_poi_ctr, self.list_poi_gmv, self.rs_mu, self.rs_lambda)
        with tf.name_scope('vvca_payment'):
            # vvca_pay [-1, 4]
            self.pay = self._create_vvca_payment(self.rs, self.top_list_idx, self.list_poi_mask_feature, N)
        with tf.name_scope('differentiable_sorting'):
            # 多序列做可微分排序
            self.Mr = self._create_differentiable_sorting(self.rs, self.top_list_idx, self.list_size, N, K)

    def _create_mu_network(self, list_poi_feat):
        cur_network_input = tf.reshape(list_poi_feat, [-1, 1, MU_INPUT_DIM])
        for i in range(0, len(MU_MLP)):
            cur_network_input = tf.add(
                tf.matmul(cur_network_input, self.mu_network_weights['layer_%d' % i]),
                self.mu_network_weights['bias_%d' % i], name='mu_network_out_%d' % i)
        out = tf.reshape(cur_network_input, [-1, MAX_LIST_NUM, PER_LIST_POI_NUM])
        return out

    def _create_lambda_network(self, list_poi_feat):
        # todo lambda只用这些特征不合理，需要优化
        cur_network_input = tf.reshape(list_poi_feat, [-1, MAX_LIST_NUM, LAMBDA_INPUT_DIM])
        for i in range(0, len(LAMBDA_MLP)):
            cur_network_input = tf.add(
                tf.matmul(cur_network_input, self.lambda_network_weights['layer_%d' % i]),
                self.lambda_network_weights['bias_%d' % i], name='lambda_network_out_%d' % i)
        out = tf.reshape(cur_network_input, [-1, MAX_LIST_NUM, PER_LIST_POI_NUM])
        return out

    def _create_vvca_payment(self, rs, top_list_idx, list_poi_mask, N):
        self.next_list_rs = tf.tile(tf.reshape(rs, [-1, 1, 1, N]), [1, N, PER_LIST_POI_NUM, 1])  # [-1, N, 4, N]
        # todo : list_poi_mask，对于list2,list3来说，都需要去除前一个，所以目前仅支持第一个list训练，如果需要多个list，需优化mask
        self.masked_next_list_rs = self.next_list_rs * list_poi_mask  # [-1, N, 4, N]
        self.except_self_item_next_max_list_rs = tf.reduce_max(self.masked_next_list_rs, axis=-1)  # [-1, N, 4]

        self.tile_list_rs = tf.tile(tf.reshape(rs, [-1, N, 1]), [1, 1, PER_LIST_POI_NUM])  # [-1, N, 4]
        self.rs_gap = self.tile_list_rs - self.except_self_item_next_max_list_rs  # [-1, N, 4]
        self.pay = self.list_poi_bid - self.rs_gap / (self.list_poi_ctr, self.rs_mu)  # [-1, N, 4]
        # todo : 做一些异常处理，截断
        # 只有第一个计费
        self.tile_top1_list_idx = tf.tile(tf.reshape(top_list_idx, [-1, 1, 1]), [1, PER_LIST_POI_NUM, 1])  # [-1, 4, 1]
        self.pay_top1 = tf.batch_gather(tf.transpose(self.pay, perm=[0, 2, 1]), self.tile_top1_list_idx)  # [-1, 4, 1]

        pay = tf.reshape(self.pay_top1, [-1, PER_LIST_POI_NUM])
        return pay

    def _create_loss(self, labels):
        N = MAX_LIST_NUM
        self.label_gmv = tf.gather(labels['gmv'], list(range(N)), axis=1)
        # 主loss分位置加权
        if USE_BIAS_WEIGHT:
            bias_weight = tf.tile(tf.reshape(tf.gather(
                self.bias_weight, list(range(PAY_LIST_NUM)), axis=1), [PAY_LIST_NUM, 1]), [1, N])
            Mr = tf.multiply(self.Mr, bias_weight)
        else:
            Mr = self.Mr
        _, reward_pay_ctr_tile = self._gen_reward(N, "pay*ctr")
        _, reward_bid_ctr_tile = self._gen_reward(N, "bid*ctr")
        _, reward_gmv_tile = self._gen_reward(N, "gmv")
        all_reward, _ = self._gen_reward(N)
        self.loss_0_pay_ctr = -LOSS_WEIGHT[0] * tf.reduce_sum(
            tf.reduce_sum(tf.multiply(Mr, reward_pay_ctr_tile), axis=[2, 1]))
        self.loss_0_bid_ctr = -LOSS_WEIGHT[0] * tf.reduce_sum(
            tf.reduce_sum(tf.multiply(Mr, reward_bid_ctr_tile), axis=[2, 1]))
        self.loss_0_gmv = -LOSS_WEIGHT[0] * tf.reduce_sum(
            tf.reduce_sum(tf.multiply(Mr, reward_gmv_tile), axis=[2, 1]))
        self.loss_1_aux = -LOSS_WEIGHT[1] * tf.reduce_sum(self._auxiliary_loss_of_reward(all_reward, PAY_LIST_NUM, N))  # [batch_size, N]
        self.loss_2_payrate = 0.0
        self.loss = self.loss_0_pay_ctr + self.loss_0_bid_ctr + self.loss_0_gmv + self.loss_1_aux + self.loss_2_payrate

    def _gen_reward(self, N, type="all"):
        if type == 'pay*ctr':
            reward = self.pay * self.list_poi_ctr
        elif type == 'bid*ctr':
            reward = self.list_poi_bid * self.list_poi_ctr
        elif type == 'gmv':
            reward = self.list_poi_ctr * self.list_poi_cvr * self.list_poi_pprice
        else:
            # todo fix reward
            reward = self.pay * self.list_poi_ctr
        # reward = self.tfPrint(reward, type)
        # reward_tile for Mr
        reward_tile = tf.reshape(tf.tile(reward, [1, PAY_LIST_NUM]), [-1, PAY_LIST_NUM, N])
        return reward, reward_tile

    def _auxiliary_loss_of_reward(self, reward, K, N):
        reward_sort = tf.math.top_k(reward, k=K)
        reward_arg_sort_idx = reward_sort.indices
        reward_sort_one_hot_matrix = tf.one_hot(reward_arg_sort_idx, N)
        # 主loss分位置加权
        if USE_BIAS_WEIGHT:
            bias_weight = tf.tile(tf.reshape(tf.gather(self.bias_weight, list(range(K)), axis=1), [K, 1]), [1, N])
            reward_sort_one_hot_matrix = tf.multiply(reward_sort_one_hot_matrix, bias_weight)
        masked_Mr = tf.log(tf.where(tf.less(self.Mr, 1e-8), tf.ones_like(self.Mr), self.Mr))
        loss_ce = tf.multiply(reward_sort_one_hot_matrix, masked_Mr)
        loss_ce = tf.reduce_sum(loss_ce, axis=[2, 1])
        return loss_ce

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
        elif 'save_model' in params.keys():
            outputs = {
                'rs_output': tf.identity(self.rs, "rs_output"),
                'pay_output': tf.identity(self.pay, "pay_output"),
                'pay_mask': tf.identity(self.pay_len_mask, "pay_mask"),
                'merge_result': tf.identity(tf.squeeze(tf.concat([tf.expand_dims(self.rs * 1000, axis=-1),
                                                                  tf.expand_dims(self.pay, axis=-1)], axis=-1),
                                                       axis=0), "merge_result")
            }
        else:
            N = MAX_LIST_NUM
            # 计算一些base的指标，用于比较
            base_rs = tf.squeeze(tf.gather(features['rs'], list(range(N)), axis=1), axis=-1)
            # base_pay_mask = tf.map_fn(self._mask_gsp_with_pay_poi_len, (base_rs, self.poi_len_mask), dtype=tf.float32)
            pay_poi_index = tf.math.top_k(base_rs, k=PAY_LIST_NUM).indices
            pay_poi_one_value = tf.cast(tf.ones_like(pay_poi_index), dtype=tf.float32)
            base_pay_mask = self._batch_scatter_nd_new(pay_poi_index, pay_poi_one_value, PAY_LIST_NUM, tf.shape(base_rs))
            base_pay_mask = tf.multiply(base_pay_mask, self.poi_len_mask)
            ctr = self.poi_ctr
            cvr = self.poi_cvr
            dna_weight_mask = self.pay_len_mask
            base_weight_mask = base_pay_mask
            if USE_BIAS_WEIGHT:
                bias_weight = tf.gather(self.bias_weight, list(range(PAY_LIST_NUM)), axis=1)
                batch_bias_weight = tf.multiply(pay_poi_one_value, bias_weight)
                dna_bias_weight = self._batch_scatter_nd_new(tf.math.top_k(self.rs, k=PAY_LIST_NUM).indices,
                                                             batch_bias_weight, PAY_LIST_NUM,
                                                             tf.shape(base_rs))
                base_bias_weight = self._batch_scatter_nd_new(pay_poi_index,
                                                              batch_bias_weight, PAY_LIST_NUM,
                                                              tf.shape(base_rs))
                dna_weight_mask = tf.multiply(dna_weight_mask, dna_bias_weight)
                base_weight_mask = tf.multiply(base_weight_mask, base_bias_weight)

            outputs = {
                'dna_rs': tf.identity(self.rs, "dna_rs"),  # [bs,400]
                'dna_price': tf.identity(self.pay, "dna_price"),  # [bs,400]
                'dna_pay_mask': tf.identity(self.pay_len_mask, "dna_pay_mask"),
                'dna_weight_mask': tf.identity(dna_weight_mask, "dna_weight_mask"),
                'dna_len': tf.identity(tf.reduce_sum(self.pay_len_mask, axis=-1)),

                'base_rs': tf.identity(base_rs, "base_rs"),  # [bs,400]
                'base_price': tf.identity(tf.gather(features['price'], list(range(N)), axis=1)),  # [bs,400]
                'base_pay_mask': tf.identity(base_pay_mask, 'base_pay_mask'),
                'base_weight_mask': tf.identity(base_weight_mask, "base_weight_mask"),
                'base_len': tf.identity(tf.reduce_sum(base_pay_mask, axis=-1)),

                'gmv': tf.identity(tf.gather(features['gmv'], list(range(N)), axis=1), 'gmv'),  # [bs,400]
                'ueq': tf.identity(tf.gather(features['ueq'], list(range(N)), axis=1), 'ueq'),
                'ctr': tf.identity(ctr, 'ctr'),
                'cvr': tf.identity(cvr, 'cvr'),
                'bid': tf.identity(self.poi_bid, 'bid')
            }

        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                              tf.estimator.export.PredictOutput(outputs)}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)



    def _build_indicator(self, labels):
        N = MAX_LIST_NUM
        poi_cnt = tf.reduce_sum(self.pay_len_mask, axis=[1, 0])
        gmv = tf.squeeze(tf.gather(labels['gmv'], list(range(N)), axis=1), axis=-1)
        ueq = tf.squeeze(tf.gather(labels['ueq'], list(range(N)), axis=1), axis=-1)
        price = tf.squeeze(tf.gather(labels['price'], list(range(N)), axis=1), axis=-1)
        ctr = self.ctr

        dna_gmv = gmv
        dna_ueq = ueq
        dna_price_origin = self.pay
        base_gmv = gmv
        base_ueq = ueq
        base_price_origin = price

        # base
        base_rs = tf.squeeze(tf.gather(labels['rs'], list(range(N)), axis=1), axis=-1)
        # base_pay_len_mask = tf.map_fn(self._mask_gsp_with_pay_poi_len, (base_rs, self.poi_len_mask), dtype=tf.float32)
        self.pay_poi_index = tf.math.top_k(base_rs, k=PAY_LIST_NUM).indices
        pay_poi_one_value = tf.cast(tf.ones_like(self.pay_poi_index), dtype=tf.float32)
        base_pay_len_mask = self._batch_scatter_nd_new(self.pay_poi_index, pay_poi_one_value, PAY_LIST_NUM,
                                                       tf.shape(base_rs))
        base_pay_len_mask = tf.multiply(base_pay_len_mask, self.poi_len_mask)

        dna_weight_mask = self.pay_len_mask
        base_weight_mask = base_pay_len_mask
        if USE_BIAS_WEIGHT:
            bias_weight = tf.gather(self.bias_weight, list(range(PAY_LIST_NUM)), axis=1)
            self.batch_bias_weight = tf.multiply(pay_poi_one_value, bias_weight)
            self.dna_bias_weight = self._batch_scatter_nd_new(tf.math.top_k(self.rs, k=PAY_LIST_NUM).indices,
                                                              self.batch_bias_weight, PAY_LIST_NUM, tf.shape(base_rs))
            self.base_bias_weight = self._batch_scatter_nd_new(self.pay_poi_index,
                                                               self.batch_bias_weight, PAY_LIST_NUM, tf.shape(base_rs))
            dna_weight_mask = tf.multiply(dna_weight_mask, self.dna_bias_weight)
            base_weight_mask = tf.multiply(base_weight_mask, self.base_bias_weight)

        dna_rs = tf.reduce_sum(tf.multiply(self.rs, dna_weight_mask), axis=[1, 0]) / poi_cnt
        dna_ctr = tf.reduce_sum(tf.multiply(ctr, dna_weight_mask), axis=[1, 0]) / poi_cnt
        dna_ecpm = tf.reduce_sum(tf.multiply(dna_price_origin * ctr, dna_weight_mask), axis=[1, 0]) / poi_cnt
        dna_price = tf.reduce_sum(tf.multiply(dna_price_origin, dna_weight_mask), axis=[1, 0]) / poi_cnt
        dna_gmv = tf.reduce_sum(tf.multiply(dna_gmv, dna_weight_mask), axis=[1, 0]) / poi_cnt
        dna_ueq = tf.reduce_sum(tf.multiply(dna_ueq, dna_weight_mask), axis=[1, 0]) / poi_cnt
        dna_bid = tf.reduce_sum(tf.multiply(self.poi_bid, self.pay_len_mask), axis=[1, 0])
        dna_pay_rate = tf.reduce_sum(tf.multiply(dna_price_origin, self.pay_len_mask), axis=[1, 0]) / dna_bid
        dna_bid_w = tf.reduce_sum(tf.multiply(self.poi_bid, dna_weight_mask), axis=[1, 0]) / poi_cnt
        dna_pay_rate_w = dna_price / dna_bid_w

        base_poi_cnt = tf.reduce_sum(base_pay_len_mask, axis=[1, 0])
        base_rs = tf.reduce_sum(tf.multiply(base_rs, base_weight_mask), axis=[1, 0]) / base_poi_cnt
        base_ctr = tf.reduce_sum(tf.multiply(ctr, base_weight_mask), axis=[1, 0]) / base_poi_cnt
        base_ecpm = tf.reduce_sum(tf.multiply(base_price_origin * ctr, base_weight_mask), axis=[1, 0]) / base_poi_cnt
        base_price = tf.reduce_sum(tf.multiply(base_price_origin, base_weight_mask), axis=[1, 0]) / base_poi_cnt
        base_gmv = tf.reduce_sum(tf.multiply(base_gmv, base_weight_mask), axis=[1, 0]) / base_poi_cnt
        base_ueq = tf.reduce_sum(tf.multiply(base_ueq, base_weight_mask), axis=[1, 0]) / base_poi_cnt
        base_bid = tf.reduce_sum(tf.multiply(self.poi_bid, base_pay_len_mask), axis=[1, 0])
        base_pay_rate = tf.reduce_sum(tf.multiply(base_price_origin, base_pay_len_mask), axis=[1, 0]) / base_bid
        base_bid_w = tf.reduce_sum(tf.multiply(self.poi_bid, base_weight_mask), axis=[1, 0]) / base_poi_cnt
        base_pay_rate_w = base_price / base_bid_w

        def formatter_log(tensors):
            log_string0 = "indicator_info: step {}:, loss={:.4f}, " \
                          "loss_0_ecpm={:.4f}, loss_0_ecpmbid={:.4f}, loss_0_gmv={:.4f}, loss_0_cvr={:.4f}, loss_1_aux={:.4f}, loss_2_payrate={:.4f}, tar={:.3f}".format(
                tensors["step"], tensors["loss"], tensors["loss_0_ecpm"], tensors["loss_0_ecpmbid"],
                tensors["loss_0_gmv"], tensors["loss_0_cvr"], tensors["loss_1_aux"],
                tensors["loss_2_payrate"], tensors["tar"]
            )

            log_string1 = "indicator_base: step {}:, avg_price={:.4f}, base_avg_price={:.4f}, pay_rate={:.4f}, base_pay_rate={:.4f}, pay_rate_w={:.4f}, base_pay_rate_w={:.4f} " \
                          "avg_rs={:.4f}, base_avg_rs={:.4f}, avg_ctr={:.4f}, base_avg_ctr={:.4f}, avg_ecpm={:.4f}, base_avg_ecpm={:.4f}," \
                          "avg_gmv={:.4f}, base_avg_gmv={:.4f}, avg_bid={:.4f}, base_avg_bid={:.4f}, avg_bid_w={:.4f}, base_avg_bid_w={:.4f}".format(
                tensors["step"], tensors["avg_price"], tensors["base_avg_price"], tensors["pay_rate"],
                tensors["base_pay_rate"],
                tensors["pay_rate_w"], tensors["base_pay_rate_w"],
                tensors["avg_rs"], tensors["base_avg_rs"], tensors["avg_ctr"], tensors["base_avg_ctr"],
                tensors["avg_ecpm"], tensors["base_avg_ecpm"], tensors["avg_gmv"], tensors["base_avg_gmv"],
                tensors["avg_bid"], tensors["base_avg_bid"],
                tensors["avg_bid_w"], tensors["base_avg_bid_w"]
            )

            log_string2 = "indicator_diff: step {}:, " \
                          "price_diff={:.3f}%, bid_diff={:.3f}%, bid_w_diff={:.3f}%, pay_rate_diff={:.3f}%, pay_rate_w_diff={:.3f}%, rs_diff={:.4f}%, " \
                          "ctr_diff={:.3f}%, ecpm_diff={:.3f}%, gmv_diff={:.3f}%".format(
                tensors["step"],
                (tensors["avg_price"] / tensors["base_avg_price"] - 1) * 100,
                (tensors["avg_bid"] / tensors["base_avg_bid"] - 1) * 100,
                (tensors["avg_bid_w"] / tensors["base_avg_bid_w"] - 1) * 100,
                (tensors["pay_rate"] / tensors["base_pay_rate"] - 1) * 100,
                (tensors["pay_rate_w"] / tensors["base_pay_rate_w"] - 1) * 100,
                (tensors["avg_rs"] / tensors["base_avg_rs"] - 1) * 100,
                (tensors["avg_ctr"] / tensors["base_avg_ctr"] - 1) * 100,
                (tensors["avg_ecpm"] / tensors["base_avg_ecpm"] - 1) * 100,
                (tensors["avg_gmv"] / tensors["base_avg_gmv"] - 1) * 100
            )
            return "\n" + log_string0 + "\n" + log_string1 + "\n" + log_string2

        self.logging_hook = tf.train.LoggingTensorHook({"loss": self.loss,
                                                        "step": tf.train.get_global_step(),
                                                        "loss_0_ecpm": self.loss_0_pay,
                                                        "loss_0_gmv": self.loss_0_gmv,
                                                        "loss_0_ecpmbid": self.loss_0_ecpm,
                                                        "loss_0_cvr": self.loss_0_cvr,
                                                        "loss_1_aux": self.loss_1_aux,
                                                        "loss_2_payrate": self.loss_2_payrate,
                                                        "tar": self.tar,
                                                        "avg_price": dna_price,
                                                        "base_avg_price": base_price,
                                                        "avg_rs": dna_rs,
                                                        "base_avg_rs": base_rs,
                                                        "pay_rate": dna_pay_rate,
                                                        "base_pay_rate": base_pay_rate,
                                                        "pay_rate_w": dna_pay_rate_w,
                                                        "base_pay_rate_w": base_pay_rate_w,
                                                        "avg_ctr": dna_ctr,
                                                        "base_avg_ctr": base_ctr,
                                                        "avg_ecpm": dna_ecpm,
                                                        "base_avg_ecpm": base_ecpm,
                                                        "avg_gmv": dna_gmv,
                                                        "base_avg_gmv": base_gmv,
                                                        "avg_bid": dna_bid,
                                                        "base_avg_bid": base_bid,
                                                        "avg_bid_w": dna_bid_w,
                                                        "base_avg_bid_w": base_bid_w,
                                                        "avg_ueq": dna_ueq,
                                                        "base_avg_ueq": base_ueq,
                                                        }, every_n_iter=10,
                                                       formatter=formatter_log)
        # 如果想看曲线，可以在这里加，然后看tensorboard
        tf.summary.scalar("loss", self.loss)

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

    def _init_list_wise_model_feature(self, features):
        # 5 [bs * list_size, poi_size, dense_feat_size]
        self.lwm_list_dense_feature = tf.gather(tf.reshape(features['list_poi_dense_feature'],
                                                           [-1, PER_LIST_POI_NUM, POI_DENSE_FEA_NUM]), list(range(5, 10)), axis=-1)
        # 6 [bs * list_size, poi_size, cate_feat_size]
        self.lwm_list_cate_feature = tf.reshape(features['list_poi_cate_feature'],
                                                [-1, PER_LIST_POI_NUM, POI_CATE_FEA_NUM])
        # 3 ctr,cvr,gmv  [bs * list_size, poi_size, pred_feat_size]
        self.lwm_list_predict_feature = tf.gather(tf.reshape(features['list_poi_dense_feature'],
                                                             [-1, PER_LIST_POI_NUM, POI_DENSE_FEA_NUM]), list(range(1, 4)), axis=-1)
        # 2 是否有铂金, 是否有合约, 全0处理 [bs * list_size, 2]
        self.lwm_dense_feature = tf.zeros([tf.shape(self.lwm_list_cate_feature)[0], 2])

    def _create_list_wise_model_weights(self):
        self.A_network_input_dim = LWP['A_NETWORK_INPUT_DIM']
        a_network_deep_layers_list = [int(item) for item in LWP['A_NETWORK_DEEP_LAYERS'].split(',')]
        self.A_network_deep_layers = a_network_deep_layers_list
        self.P_network_input_dim = LWP['P_NETWORK_INPUT_DIM']
        p_network_deep_layers_list = [int(item) for item in LWP['P_NETWORK_DEEP_LAYERS'].split(',')]
        self.P_network_deep_layers = p_network_deep_layers_list
        # feature_emb_weights 和vvca用同一个
        with tf.name_scope('A_network_weights'):
            self.A_network_weights = self._create_network_weights(self.A_network_deep_layers,
                                                                  self.A_network_input_dim,
                                                                  "A", trainable=False)
        with tf.name_scope('P_network_weights'):
            self.P_network_weights = self._create_network_weights(self.P_network_deep_layers,
                                                                  self.P_network_input_dim,
                                                                  "P", trainable=False)
        pass

    def _init_list_wise_weights_from_checkpoint(self):
        assignment_map = {
            'cate_feature_embeddings': 'cate_feature_embeddings'
            , 'dense/': 'dense/'
        }
        weight_prefix, network_num_layer = "A", len(self.A_network_deep_layers)
        for i in range(0, network_num_layer):
            assignment_map['%s_layer_%d' % (weight_prefix, i)] = '%s_layer_%d' % (weight_prefix, i)
            assignment_map['%s_bias_%d' % (weight_prefix, i)] = '%s_bias_%d' % (weight_prefix, i)
        weight_prefix, network_num_layer = "P", len(self.P_network_deep_layers)
        for i in range(0, network_num_layer):
            assignment_map['%s_layer_%d' % (weight_prefix, i)] = '%s_layer_%d' % (weight_prefix, i)
            assignment_map['%s_bias_%d' % (weight_prefix, i)] = '%s_bias_%d' % (weight_prefix, i)
        tf.train.init_from_checkpoint(LWP['CHECKPOINT_PATH'], assignment_map)

    def _create_list_wise_model(self):
        with tf.name_scope('dnn_model'):
            with tf.name_scope('P_network'):
                screen_cate_features = self.lwm_list_cate_feature
                screen_cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
                    self.feature_emb_weights['cate_feature_embeddings'], screen_cate_features),
                    [-1, LWP['LIMIT_NUM'], LWP['POI_CATE_FEA_NUM'] * LWP['CATE_EMBEDDING_SIZE']])
                P_network_input_screen = tf.reshape(tf.concat([self.lwm_list_dense_feature, screen_cate_feature_embeddings], axis=2, name='P_network_input_screen'),
                                                    [-1, LWP['P_NETWORK_INPUT_DIM']])
                p_network_output = self._create_p_network(P_network_input_screen)
            with tf.name_scope('A_network'):
                poi_predict_fea = tf.reshape(self.lwm_list_predict_feature, [-1, LWP['POI_PREDICT_FEA_NUM']])
                p_network_output = tf.concat([p_network_output, poi_predict_fea], axis=1)
                P_network_output_screen = tf.reshape(p_network_output,
                                                     [-1, LWP['LIMIT_NUM'] * (LWP['POI_P_PREDICT_OUT_NUM'] + LWP['POI_PREDICT_FEA_NUM'])])
                A_network_input = tf.concat([self.lwm_dense_feature, P_network_output_screen], axis=1)
                A_network_output = self._create_a_network(A_network_input)
            return A_network_output

    def _create_p_network(self, p_network_input):
        cur_network_input = p_network_input
        for i in range(0, len(self.P_network_deep_layers)):
            cur_network_input = tf.add(
                tf.matmul(cur_network_input, self.P_network_weights['layer_%d' % i]),
                self.P_network_weights['bias_%d' % i], name='p_network_out_%d' % i)
            cur_network_input = tf.nn.swish(cur_network_input)
        out = cur_network_input
        return out

    def _create_a_network(self, a_network_input):
        cur_network_input = a_network_input
        for i in range(0, len(self.A_network_deep_layers)):
            cur_network_input = tf.add(tf.matmul(cur_network_input, self.A_network_weights['layer_%d' % i]),
                                       self.A_network_weights['bias_%d' % i], name='a_network_out_%d' % i)
            cur_network_input = tf.nn.swish(cur_network_input)
        out = tf.layers.dense(cur_network_input, LWP['LIMIT_NUM'], trainable=False)
        out = tf.sigmoid(out)
        return out