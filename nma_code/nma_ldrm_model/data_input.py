# -*- coding: utf-8 -*-
import tensorflow as tf
from config import *
import numpy as np
from tools import tick_tock


def generate_parse_tfrecord_local_fn():
    def _parse_function(batch_examples):
        common_features, sequence_features = feature_parse_scheme()
        parsed_features = tf.parse_example(
            serialized=batch_examples,
            features=common_features
        )
        features = feature_product(parsed_features)
        labels = label_product(parsed_features)
        return features, labels

    return _parse_function


def generate_parse_valid_tfrecord_local_fn():
    def _parse_function(batch_examples):
        common_features, sequence_features = feature_parse_scheme()
        parsed_features = tf.parse_example(
            serialized=batch_examples,
            features=common_features
        )
        features = feature_product(parsed_features)
        labels = label_product(parsed_features)
        return features, labels

    return _parse_function


def feature_parse_scheme():
    label_len = 0 
    feature_len = POI_FEA_NUM * POI_NUM + 1 + POI_FEA_NUM * SUB_POI_NUM

    common_features = {
        "label": tf.FixedLenFeature([label_len], dtype=tf.float32),
        "feature": tf.FixedLenFeature([feature_len], dtype=tf.float32),
    }

    sequence_features = {}
    return common_features, sequence_features


def label_product(parsed_features):
    label_buffer = parsed_features['label']
    labels_result = {}
    return labels_result


def feature_product(parsed_features):
    feature_buffer = parsed_features['feature']
    poi_features_start = 0
    poi_features_end = poi_features_start + POI_FEA_NUM * POI_NUM
    poi_size_feature_start = poi_features_end
    poi_size_feature_end = poi_size_feature_start + 1
    poi_sub_features_start = poi_size_feature_end
    poi_sub_features_end = poi_sub_features_start + POI_FEA_NUM * SUB_POI_NUM


    poi_dense_feature = tf.reshape(
        tf.gather(feature_buffer, list(range(poi_features_start, poi_features_end)), axis=1),
        [-1, POI_NUM, POI_FEA_NUM])
    # poi_bid = tf.gather(poi_dense_feature, list(range(0, 1)), axis=2)
    # poi_ctr = tf.gather(poi_dense_feature, list(range(, 5)), axis=2)/1e6
    poi_size = tf.gather(feature_buffer, list(range(poi_size_feature_start, poi_size_feature_end)), axis=1)

    poi_sub_feature = tf.reshape(
        tf.gather(feature_buffer, list(range(poi_sub_features_start, poi_sub_features_end)), axis=1),
        [-1, SUB_POI_NUM, POI_FEA_NUM])

    poi_feature_shape = tf.shape(poi_dense_feature)
    bid = tf.random.uniform(shape=[poi_feature_shape[0], poi_feature_shape[1]], minval=0.5, maxval=1.5, seed=2023)
    ctr = tf.gather(poi_dense_feature, list(range(1, 2)), axis=2)
    features_result = {
        'poi_dense_feature': poi_dense_feature,
        'poi_sub_feature': poi_sub_feature,
        'poi_size': poi_size,
        'bid': bid,
        'ctr': ctr
    }


    return features_result


# num_parallel 表示cpu的核数,用于控制 map的并行度
def input_fn_maker(file_names, is_train, batch_size, epoch=None, num_parallel=4):
    def input_fn():
        _parse_fn = generate_parse_tfrecord_local_fn() if is_train else generate_parse_valid_tfrecord_local_fn()
        files = tf.data.Dataset.list_files(file_names)
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4 * 10))
        dataset = dataset.prefetch(buffer_size=batch_size * 10)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_parse_fn, num_parallel_calls=num_parallel)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn


# 从hive表统计得到均值和方差文件
def get_normalization_parameter(mean_var_path):
    with tf.gfile.Open(mean_var_path) as f:
        fea_mean = f.readline().strip().split('\t')
        fea_var = f.readline().strip().split('\t')
        cont_fea_mean = map(float, fea_mean)
        cont_fea_var = map(float, fea_var)
    f.close()
    return cont_fea_mean, cont_fea_var


def get_bias_weight_parameter(bias_weight_path):
    with tf.gfile.Open(bias_weight_path) as f2:
        fea_mean = f2.readline().strip().split('\t')
        cont_fea_mean = map(float, fea_mean)
    f2.close()
    return cont_fea_mean


if __name__ == '__main__':
    # train_file = TRAIN_FILE
    train_file = ["/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_dataset/avito_v1_simulate_for_DNA/train_data/part-r-00041"]
    train_input_fn = input_fn_maker(train_file, is_train=True, batch_size=10, epoch=1)
    features, labels = train_input_fn()

    sess = tf.Session()
    try:
        with tick_tock("DATA_INPUT") as _:
            features_np, labels_np = sess.run([features, labels])
        for key in features_np:
            print("=" * 50, key, np.shape(features_np[key]))
            print(features_np[key])

        for key in labels_np:
            print("=" * 50, key, np.shape(labels_np[key]))
            print(labels_np[key])

    except Exception as e:
        print(e)
