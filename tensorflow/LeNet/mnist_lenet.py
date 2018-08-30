# -*- coding: UTF-8 -*-
import os

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# LeNet-5 各层参数定义
C1_KERNEL_SIZE = 5
C1_KERNEL_DEPTH = 6
C1_STRIDES = [1, 1, 1, 1]

S1_POOL_SIZE = [1, 2, 2, 1]
S1_STRIDES = [1, 2, 2, 1]

C3_KERNEL_SIZE = 5
C3_KERNEL_DEPTH = 16
C3_STRIDES = [1, 1, 1, 1]

S4_POOL_SIZE = [1, 2, 2, 1]
S4_STRIDES = [1, 2, 2, 1]

C5_KERNEL_SIZE = 5
C5_KERNEL_DEPTH = 120
C5_STRIDES = [1, 1, 1, 1]

F6_NODES = 84
F7_NODES = 10

IMAGE_SIZE = 32
IMAGE_CHANNELS = 1
NUM_LABELS = 10

# 神经网络配置参数
BATCH_SIZE = 100  # 小批量梯度下降每个batch的样本数
EPOCH = 100  # 训练轮数

LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNIING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 损失函数中模型参数正则化项的权重
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 模型保存路径与文件名
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"


def inference(input_tensor, train=True, regularizer=None):
    reuse = not train
    with tf.variable_scope("layer1-C1", reuse=reuse):
        weights = tf.get_variable("weights",
                                  [C1_KERNEL_SIZE, C1_KERNEL_SIZE, IMAGE_CHANNELS, C1_KERNEL_DEPTH],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [C1_KERNEL_DEPTH], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, weights, C1_STRIDES, padding="VALID")
        layer1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))

    with tf.variable_scope("layer2-S2", reuse=reuse):
        layer2 = tf.nn.max_pool(layer1, ksize=S1_POOL_SIZE, strides=S1_STRIDES, padding="SAME")

    with tf.variable_scope("layer3-C3", reuse=reuse):
        weights = tf.get_variable("weights",
                                  [C3_KERNEL_SIZE, C3_KERNEL_SIZE, C1_KERNEL_DEPTH, C3_KERNEL_DEPTH],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [C3_KERNEL_DEPTH], initializer=tf.constant_initializer(0.0))

        conv3 = tf.nn.conv2d(layer2, weights, C3_STRIDES, padding="VALID")
        layer3 = tf.nn.relu(tf.nn.bias_add(conv3, biases))

    with tf.variable_scope("layer4-S4", reuse=reuse):
        layer4 = tf.nn.max_pool(layer3, ksize=S4_POOL_SIZE, strides=S4_POOL_SIZE, padding="SAME")

    fc = tf.contrib.layers.flatten(layer4)
    with tf.variable_scope("layer5-C5", reuse=reuse):
        weights = tf.get_variable("weights",
                                  [C5_KERNEL_SIZE * C5_KERNEL_SIZE * C3_KERNEL_DEPTH, C5_KERNEL_DEPTH],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [C5_KERNEL_DEPTH], initializer=tf.constant_initializer(0.0))

        layer5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc, weights), biases))

    with tf.variable_scope("layer6-FC6", reuse=reuse):
        weights = tf.get_variable("weights",
                                  [C5_KERNEL_DEPTH, F6_NODES],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [F6_NODES], initializer=tf.constant_initializer(0.0))
        layer6 = tf.nn.relu(tf.matmul(layer5, weights) + biases)
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weights))
        if train:
            layer6 = tf.nn.dropout(layer6, 0.5)

    with tf.variable_scope("layer7-FC7", reuse=reuse):
        weights = tf.get_variable("weights",
                                  [F6_NODES, F7_NODES],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [F7_NODES], initializer=tf.constant_initializer(0.0))
        output = tf.nn.relu(tf.matmul(layer6, weights) + biases)
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weights))

    return output


def do_eval(sess, xs, ys):
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS], name="x-input")
    y = tf.placeholder(tf.float32, [None, NUM_LABELS], name="y-input")

    y_halt = inference(x, train=False)

    # 定义损失函数为交叉熵.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_halt, labels=y)

    # 计算当前batch中所有样本的交叉熵均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 验证集上精度
    correct_prediction = tf.equal(tf.argmax(y_halt, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss, accuracy_score = sess.run([loss, accuracy], feed_dict={x: xs, y: ys})
    return loss, accuracy_score


def do_train(data):
    '''
    模型训练
    :param mnist:
    :return:
    '''
    train_graph = tf.Graph()

    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS], name="x-input")
    y = tf.placeholder(tf.float32, [None, NUM_LABELS], name="y-input")

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算在当前参数下网络前向传播的结果。这里不使用滑动平均
    y_halt = inference(x, regularizer=regularizer)

    # 定义存储训练轮数的变量，并将其指定为不可训练的。在使用TensorFlow训练神经网络时，一般将代表轮数的变量指定为不可训练
    global_step = tf.Variable(0, trainable=False)

    # 用滑动平均衰减率和训练轮数变量初始化滑动平均类。给定训练轮数可以加快训练早期变量更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 通过tf.trainable_variables()返回计算图上的可训练变量集合（GraphKeys.TRAINABLE_VARIABLES）中的元素，这个集合中的
    # 元素就是所有trainable=False的变量。对这可训练变量使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 定以损失函数为交叉熵.
    # tensorflow中提供了sparse_softmax_cross_entropy_with_logits函数计算交叉熵
    # 这个函数的第logits参数是神经网络不包括softmax层的前向传播结果,labels参数是训练数据的标注
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_halt, labels=y)

    # 计算当前batch中所有样本的交叉熵均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率,学习率在此基础上随着迭代次数而减小
        global_step,  # 当前迭代的次数
        data.train.num_examples / BATCH_SIZE,  # 一轮迭代所需要的迭代次数
        LEARNIING_RATE_DECAY  # 学习率的衰减率
    )

    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时,每过一遍数据既需要通过反向传播算法来跟新神经网络中的参数,又要更新每一个参数的滑动平均值.
    # tensorflow中提供了tf.cttronl_dependencies和tf.group两种机制实现一次完成多个操作.
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化持久化类
    saver = tf.train.Saver()

    # 初始化会话并开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 验证集数据
        xs_validation = np.pad(data.validation.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        ys_validation = data.validation.labels

        # 迭代训练神经网络
        steps = 0
        while data.train.epochs_completed < EPOCH:
            # 产生这一轮使用的一个batch数据,并进行训练
            xs, ys = data.train.next_batch(BATCH_SIZE)
            xs = np.pad(xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            sess.run(train_step, feed_dict={x: xs, y: ys})
            steps += 1
            # 每迭代10轮，输出保存一次模型
            if steps % 100 == 0:
                loss_train, accuracy_train = do_eval(sess, xs, ys)
                loss_validation, accuracy_validation = do_eval(sess, xs_validation, ys_validation)

                print(
                    "After %d training step(s),loss on training batch is %g, accuray on training bath is %g; loss on validation is %g, accuracy on validation is %g." % (
                        steps, loss_train, accuracy_train, loss_validation, accuracy_validation))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        # 测试集
        xs_test = np.pad(data.test.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        ys_test = data.test.labels
        _, test_acc = do_eval(sess, xs_test, ys_test)
        print("After %d training step(s),test accuracy using average model is %g" % (steps, test_acc))


def main(argv=None):
    '''
    主程序入口
    :param argv:
    :return:
    '''
    # 加载MNSIT数据集
    mnist = input_data.read_data_sets("../mnist/data", reshape=False, one_hot=True)
    do_train(mnist)


if __name__ == '__main__':
    tf.app.run()  # 会自动调用main函数
