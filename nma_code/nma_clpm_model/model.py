# coding: utf-8 -*-
from data_input import *
from tools import *
from config import *
from util import *
import numpy as np
import tensorflow as tf
from layers import *
import os
from collections import namedtuple
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DNN:
    def __init__(self):
        pass

    # env_feature = > dense_feature
    # cxr_feature = > screen_predict_feature
    # cat_feature = > screen_cate_feature
    # dense_feature = > screen_dense_feature


    def _build_atten_layer(self, features):
        # nature poi做self-attention
        mask = tf.squeeze(tf.sequence_mask(features["nature_poi_num"], FEATURE_NATURE_POI))
        nature_poi = multihead_attention(queries=self.nature_poi,
                                         keys=self.nature_poi,
                                         values=self.nature_poi,
                                         key_masks=mask,
                                         num_heads=4,
                                         dropout_rate=0.1,
                                         training=self.train
                                         )

        nature_poi = feedforward(nature_poi,
                              num_units=[CATE_FEATURE_EMBEDDINGS_SHAPE[-1],
                                         CATE_FEATURE_EMBEDDINGS_SHAPE[-1]],
                              is_training=self.train)

        # 与ad_poi做target-attention
        target_att = multihead_attention(queries=self.ad_poi,
                                         keys=nature_poi,
                                         values=nature_poi,
                                         key_masks=mask,
                                         num_heads=4,
                                         dropout_rate=0.1,
                                         training=self.train,
                                         causality=True,
                                         scope="target_att"
                                         )

        target_att = feedforward(target_att,
                                 num_units=[CATE_FEATURE_EMBEDDINGS_SHAPE[-1],
                                            CATE_FEATURE_EMBEDDINGS_SHAPE[-1]],
                                 is_training=self.train,
                                 scope="target_att"
                                 )
        return target_att

    def _build_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            nature_embedding = self._build_atten_layer(features)
            fc_out = tf.concat([self.input_embedding, nature_embedding], axis=2) # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None, name=dense_name)
                fc_out = tf.nn.swish(fc_out)

            fc_out = tf.concat([fc_out], axis=2)
            fc_out_ctr, fc_out_imp = tf.reshape(fc_out, [-1, A_INPUT_DIM]), tf.reshape(fc_out, [-1, A_INPUT_DIM])
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                fc_out_ctr = tf.nn.swish(fc_out_ctr)
            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="final_out_ctr")
            self.ctr = ctr_out 
            if not self.train:
                ctr_out = tf.nn.sigmoid(ctr_out)

            self.out = tf.concat([ctr_out], axis=-1)
            self.Q_network_output = self.out


    def _create_loss(self, labels):
        with tf.name_scope('loss'):
            self.label = labels['ctr_label']
            self.mask = labels['mask']
            # ctr_loss
            self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.out, weights=self.mask))

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
    
    def _create_weights(self):
        with tf.name_scope('feature_emb_weights'):
            self.feature_weights = {
                'embedding': tf.get_variable('embedding',
                                             shape=CATE_FEATURE_EMBEDDINGS_SHAPE,
                                             initializer=tf.zeros_initializer())
            }

    def _process_features(self, features):
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature

        # N * M * K
        # N * D ( D <= M )
        self.cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], features['cate_feature']),
            [-1, POI_NUM, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.nature_poi = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], features['nature_poi']),
            [-1, FEATURE_NATURE_POI, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.ad_poi = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], features['ad_poi']),
            [-1, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) 
        self.input_embedding = tf.concat(
            [self.cate_feature_embeddings], axis=2)
        self.feat_predict = features['dense_feature']
        return features

    def _get_attr_hash(self, tensors, emb_table, num):
        first_cate, second_cate, thrid_cate, _ = tf.split(tensors,[1, 1, 1, 3], axis=2)
        first_cate, second_cate, thrid_cate  = tf.squeeze(first_cate, axis=2), tf.squeeze(second_cate, axis=2),  tf.squeeze(thrid_cate, axis=2)
        first_cate = tf.reshape(tf.nn.embedding_lookup(
            emb_table, first_cate),
            [-1, num, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        second_cate = tf.reshape(tf.nn.embedding_lookup(
            emb_table, second_cate),
            [-1, num, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        thrid_cate = tf.reshape(tf.nn.embedding_lookup(
            emb_table, thrid_cate),
            [-1, num, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        return first_cate, second_cate, thrid_cate


    def _delivery_hash(self, tensors):
        feat_fei, feat_juli, feat_shijian, feat_qisongjia = tf.split(tensors, [1, 1, 1, 1], axis=2)
        feat_fei, feat_juli, feat_shijian, feat_qisongjia = float_custom_hash(feat_fei, "feat_fei", 1),  float_custom_hash(feat_juli, "feat_juli"), float_custom_hash(feat_shijian, "feat_shijian"), float_custom_hash(feat_qisongjia, "feat_qisongjia", 2)
        ad_delivery = tf.concat([feat_fei, feat_juli, feat_shijian, feat_qisongjia], axis=-1)
        return ad_delivery

    def _create_indicator(self, labels):
        ctr_out = self.out
        ctr_out = tf.reduce_mean(tf.nn.sigmoid(ctr_out))

        # All gradients of loss function wrt trainable variables
        '''
        grads = tf.gradients(self.loss, tf.trainable_variables())
        for grad, var in list(zip(grads, tf.trainable_variables())):
            tf.summary.histogram(var.name + '/gradient', grad)
        '''

        def format_log(tensors):
            log0 = "train info: step {}, loss={:.4f}, ctr_loss={:.4f}, " \
                   "ctr_out={:.4f}".format(
                tensors["step"], tensors["loss"], tensors["loss"],
                tensors["ctr_out"],
            )
            return log0

        self.logging_hook = tf.train.LoggingTensorHook({"step": tf.train.get_global_step(),
                                                        "loss": self.loss,
                                                        "ctr_out": ctr_out 
                                                        },
                                                       every_n_iter=5,
                                                       formatter=format_log)

    def model_fn_estimator(self, features, labels, mode, params):
        self._create_weights()
        self._process_features(features)
        # self._create_list_wise_evaluate_model(features, mode)
        self._build_model(features, labels, mode, params)
        if self.train:
            self._create_loss(labels)
            self._create_optimizer()
            self._create_indicator(labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op, training_hooks=[self.logging_hook])
        else:
            if 'save_model' in list(params.keys()):
                outputs = {
                    "Q_network_output": tf.identity(self.Q_network_output, "Q_network_output"),
                    "out": tf.identity(self.out, "out")
                    }
            else:
                ctr_out = self.out
                ctr = self.ctr 
                loss = tf.losses.sigmoid_cross_entropy(features['ctr_label'], ctr, weights=features['mask'])
                # gmv
                outputs = {'out': self.out,
                           'mask': features['mask'],
                           'ctr_out': ctr_out,
                           'ctr_label': features['ctr_label'],
                           'q_out': self.Q_network_output,
                           'cxr_feature': features['dense_feature'],
                           'loss': loss
                            }
            export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                      tf.estimator.export.PredictOutput(outputs)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)


    def tf_print(self, var, varStr='null'):
        return tf.Print(var, [var], message=varStr, summarize=100)



    def _create_list_wise_evaluate_model(self, features, mode):
        self._create_list_model_weights()
        self._create_list_model_input(features)
        self._create_list_model(mode)
        self._init_from_checkpoint()


    def _create_list_model_weights(self):
        with tf.name_scope('feature_emb_weights'):
            self.feature_weights = {
                'BASE_embedding': tf.get_variable('BASE_embedding',
                                             shape=CATE_FEATURE_EMBEDDINGS_SHAPE,
                                             initializer=tf.zeros_initializer())
            }


    def _create_list_model_input(self, features):
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature

        # N * M * K
        # N * D ( D <= M )
        self.list_model_input_embedding = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['BASE_embedding'], features['cate_feature']),
            [-1, POI_NUM, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.list_model_input_dense = features['dense_feature']
        return features

    def _create_list_model(self, mode):
        self.list_model_train = (mode == tf.estimator.ModeKeys.TRAIN)
        with tf.name_scope('dnn_model'):

            fc_out = tf.concat([self.list_model_input_embedding, self.list_model_input_dense], axis=2)  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "BASE_MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                fc_out = tf.nn.swish(fc_out)
            ctr_out = tf.squeeze(tf.layers.dense(fc_out, 1, activation=None, name="BASE_final_out_ctr"), axis=2)
            ctr_out = tf.nn.sigmoid(ctr_out)
            ctr_out = tf.expand_dims(ctr_out, axis=2)
            self.list_model_out = tf.concat([ctr_out], axis=-1)
            # self.list_model_out = tf.Print(self.list_model_out, [self.list_model_out], message="list_model_out", summarize=100)
            self.list_model_out = tf.stop_gradient(self.list_model_out)

    def _init_from_checkpoint(self):
        # 注意：name_scope对于get_variable是无效的，因此不能使用scopename，需要按照tensorname进行赋值
        assignment_map = {
            'BASE_embedding': 'BASE_embedding',
            'BASE_final_out_ctr/': 'BASE_final_out_ctr/',
            'BASE_final_out_imp/': 'BASE_final_out_imp/',
            'BASE_MLP_A0/': 'BASE_MLP_A0/',
            'BASE_MLP_A1/': 'BASE_MLP_A1/',
            'BASE_MLP_A2/': 'BASE_MLP_A2/'
        }
        variable_names = [v.name for v in tf.trainable_variables()]
        print(variable_names)

        reader = tf.train.NewCheckpointReader(LIST_MODEL_CHECKPOINT_PATH)
        var_to_shape_map = reader.get_variable_to_shape_map()
        print("-------" * 10)
        for key in var_to_shape_map:
            print(key)
        tf.train.init_from_checkpoint(LIST_MODEL_CHECKPOINT_PATH, assignment_map)

