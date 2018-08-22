#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-21 下午9:22
# @Author  : yinwb
# @File    : TransferLearning.py

import os
import glob
import random

import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np

# Inception-v3模型瓶颈层的节点数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型代表瓶层结果的张量名称
BOTTLENECK_TENSOR_NAME = "pool_3/_reshape:0"

# 图片输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = "DecodeJpeg/contents:0"

# 预训练好的Inception-v3模型
MODEL_FILE = "./data/inception_dec_2015/tensorflow_inception_graph.pb"

# 缓存目录
CACHE_DIR = "./data/cache"

# 图片目录
INPUT_DATA = "./data/flower_photos"

# 验证数据的百分比
VALIDATION_PERCENTAGE = 10

# 测试数据的百分比
TESTING_PERCENTAGE = 10

# 神经网络设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 10


def create_image_lists(testing_percentage, validation_percentage):
    '''
    从数据文件夹中读取所有图片列表并按训练集/验证集和测试集分开
    :param testing_percentage: 测试集大小
    :param validation_percentage: 验证集大小
    :return: 字典
    '''
    result = {}

    sub_dirs = os.listdir(INPUT_DATA)

    for dir_name in sub_dirs:
        if os.path.isdir(os.path.join(INPUT_DATA,dir_name)):
            file_glob = os.path.join(INPUT_DATA, dir_name, "*.jpg")
            pathnames = glob.glob(file_glob)

            label_name = dir_name.lower()

            # 初始化当前类别的训练数据集/测试集和验证集
            training_images = []
            testing_images = []
            validation_images = []

            for pathname in pathnames:
                base_name = os.path.basename(pathname)
                # 将数据随机分到训练集/测试集和验证集
                random = np.random.randint(100)
                if random < VALIDATION_PERCENTAGE:
                    validation_images.append(base_name)
                elif random < TESTING_PERCENTAGE + VALIDATION_PERCENTAGE:
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)

            # 将当前类别数据存放在词典中
            result[label_name] = {
                "dir": dir_name,
                "training": training_images,
                "testing": testing_images,
                "validation": validation_images
            }
    return result


def get_image_path(image_dicts, image_dir, label_name, index, category):
    '''
    通过类别名称/所属数据集和图片编号获取一张图片的地址
    :param image_dicts: 所有图片信息
    :param image_dir: 根目录
    :param label_name: 类别名称
    :param index: 图片编号
    :param category: 验证集/测试集/训练集
    :return: 完整的文件路径
    '''
    # 获取给定类别中所有图片信息
    label_dict = image_dicts[label_name]
    # 根据所属数据集的名称获取集合中的全部图片信息
    image_lists = label_dict[category]
    mod_index = index % len(image_lists)
    # 获取图片文件名
    base_name = image_lists[mod_index]
    sub_dir = label_dict['dir']

    pathname = os.path.join(image_dir, sub_dir, base_name)
    return pathname


def get_bottleneck_path(image_dicts, label_name, index, category):
    '''
    通过类别名称/所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量文件地址
    :param image_dicts:
    :param label_name:
    :param index:
    :param category:
    :return:
    '''
    return get_image_path(image_dicts, CACHE_DIR, label_name, index, category) + ".txt"


def run_blttleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    '''
    使用加载的训练好的的Inception-v3模型处理一张图片,得到这个图片的特征向量
    :param sess:
    :param image_data:
    :param image_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''

    bottlenect_values = sess.run(bottleneck_tensor, {
        image_data_tensor: image_data
    })
    # 经过卷积神经网络处理的结果是一个四维数据,需要将这个结果压缩成一个特征向量
    bottlenect_values = np.squeeze(bottlenect_values)
    return bottlenect_values


def get_or_create_bottleneck(sess, image_dicts, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    '''
    获取一张图片经过Inception-v3模型处理后的特征向量.首先会试图从文件中加载特征向量,如果该文件不存在则计算这个特征向量并保存至文件
    :param sess:
    :param image_dicts:
    :param label_name:
    :param index:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''

    # 获取特征向量文件路径
    label_dict = image_dicts[label_name]
    sub_dir = label_dict["dir"]
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_dicts, label_name, index, category)

    # 如果特征文件不存在,则计算并保存至文件
    if not os.path.exists(bottleneck_path):
        # 获取原始图片路径
        image_path = get_image_path(image_dicts, INPUT_DATA, label_name, index, category)
        # 获取图片内容
        image_data = gfile.FastGFile(image_path, "rb").read()
        # 计算特征向量
        bottleneck_values = run_blttleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存入文件
        bottleneck_string = ",".join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, "w") as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中读取特征向量
        with open(bottleneck_path, "r") as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(",")]

    return bottleneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_dicts, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    '''
    随机获取一个batch的图片作为训练数据
    :param sess:
    :param n_classes:
    :param image_dicts:
    :param how_many:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_dicts.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_dicts, label_name, image_index, category, jpeg_data_tensor,
                                              bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0

        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def get_test_bottlenecks(sess, image_dicts, n_classes, jpeg_data_tensor, bottleneck_tensor):
    '''
    获取全部的测试集.在最终测试的时候需要使用所有的测试数据计算正确率
    :param sess:
    :param image_dicts:
    :param n_classes:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    bottlenecks = []
    ground_truths = []
    label_names = list(image_dicts.keys())
    for label_index, label_name in enumerate(label_names):
        category = "testing"
        for index, unused_base_name in enumerate(image_dicts[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_dicts, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0

            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def main(args):
    # 读取所有的图片
    image_dicts = create_image_lists(TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_dicts.keys())
    # 加载Inception-v3模型
    with gfile.FastGFile(MODEL_FILE, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                          JPEG_DATA_TENSOR_NAME])
    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name="BottleneckInputPlaceholder")
    # 定义新的神经网络输出
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name="GroundTruthInput")
    # 定义一层全连接层来解决新的图片分类问题
    with tf.name_scope("final_training_ops"):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # 训练过程
        for i in range(STEPS):
            train_bottlenecks, train_ground_truths = \
                get_random_cached_bottlenecks(sess, n_classes, image_dicts, BATCH, "training", jpeg_data_tensor,
                                              bottleneck_tensor)
            sess.run(train_step, feed_dict={
                bottleneck_input: train_bottlenecks,
                ground_truth_input: train_ground_truths
            })

            # 在验证数据上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truths = \
                    get_random_cached_bottlenecks(sess, n_classes, image_dicts, BATCH, "validation", jpeg_data_tensor,
                                                  bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks,
                    ground_truth_input: validation_ground_truths
                })

                print("Step %d: Validation accuracy on random sample %d examples = %.lf%%" % (
                    i, BATCH, validation_accuracy * 100))

        # 在测试数据上测试正确率
        test_bottlenecks, test_ground_truths = get_test_bottlenecks(sess, image_dicts, n_classes, jpeg_data_tensor,
                                                                    bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,feed_dict={
            bottleneck_input: test_bottlenecks,
            ground_truth_input: test_ground_truths
        })

        print("Final test accuracy  = %.lf%%" % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
