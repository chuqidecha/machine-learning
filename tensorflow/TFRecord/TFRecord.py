#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-17 下午9:03
# @Author  : yinwb
# @File    : TFRecord.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np


def int64_feature(value):
    '''
    整型属性
    :param value:
    :return:
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    '''
    字节型属性
    :param value:
    :return:
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_as_tf_record(datas, labels, pathname):
    '''
    将数据保存为TFRecord格式
    :param datas:
    :param labels:
    :param pathname:
    :return:
    '''
    with tf.python_io.TFRecordWriter(pathname) as writer:
        # 向量长度
        length = datas.shape[1]

        for index, data in enumerate(datas):
            # 将图像转换成一个字符串
            data_raw = data.tostring()
            # 将一个图像转换成Example Protocol Buffer,并将所有的信息写入这个数据结构
            example = tf.train.Example(features=tf.train.Features(feature={
                "length": int64_feature(length),
                "label": int64_feature(np.argmax(labels[index])),
                "data": bytes_feature(data_raw)
            }))
            # 将一个Example写入TFRecord文件
            writer.write(example.SerializeToString())


def load_from_tf_record(pathname):
    '''
    从TFRecord中加载数据
    :param pathname:
    :return:
    '''

    reader = tf.TFRecordReader()
    # 创建一个文件来维护输入文件列表
    filename_queue = tf.train.string_input_producer([pathname])

    # 从文件中读出一个样例,也可以使用read_up_to函数一次性读取多个样例
    _, serilized_example = reader.read(filename_queue)

    # 解析读出的一个样例.如果需要解析多个样例,也可以使用parse_example函数
    features = tf.parse_single_example(
        serilized_example,
        features={
            # TensorFlow中提供了两种不同的属性解析方法
            # tf.FixedLenFeature类解析的结果为一个Tensor
            # tf.VarLenFeature类即系的结果为SparseTensor,用于处理稀疏数据
            # 数据解析方式需要和写入方式保持一致
            "length": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64),
            "data": tf.FixedLenFeature([], tf.string)

        })
    # td.decode_raw可以将字符串解析成对应的像素数组
    datas = tf.decode_raw(features["data"], tf.uint8)
    labels = tf.cast(features["label"], tf.int32)
    lengths = tf.cast(features["length"], tf.int32)

    with tf.Session() as sess:
        # 启动多线程处理
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 每次读取一个样例,所有样例读完之后会从头读取
        for i in range(10):
            data, length, label = sess.run([datas, lengths, labels])
            print("data={0},length={1},label={2}".format(data[:10], length, label))


if __name__ == "__main__":
    mnist = input_data.read_data_sets("../mnist/data", dtype=tf.uint8, one_hot=True)
    images = mnist.train.images
    labels = mnist.train.labels
    save_as_tf_record(images, labels, "./mnist.tfrecords")

    load_from_tf_record("./mnist.tfrecords")
