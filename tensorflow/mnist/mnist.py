# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 神经网络配置参数
INPUT_NODE = 784  # 神经网络输入层结点数，对应于MNIST数据集中每幅图像的像素个数
OUTPUT_NODE = 10  # 神经网络输出结点数，对应于MNIST数据集类别数
LAYER1_NODE = 500  # 神经网络第一个隐层结点数
BATCH_SIZE = 100  # 小批量梯度下降每个batch的样本数

LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNIING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 损失函数中模型参数正则化项的权重
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input, weights1, biases1, weights2, biases2, avg_class=None):
    '''
    前向传播。该函数定义了含有一个隐藏的网络，并且使用ReLU激活函数实现去线性化。通过参数avg_class控制是否使用滑动平均。
    :param input:
    :param weights1:
    :param biases1:
    :param weights2:
    :param biases2:
    :param avg_class:
    :return:
    '''
    output = None
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input, weights1) + biases1)
        output = tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input, avg_class.average(weights1)) + avg_class.average(biases1))
        output = tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
    return output


def train(mnist):
    '''
    模型训练
    :param mnist:
    :return:
    '''
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    # 隐层参数初始化
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 输出层参数初始化
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下网络前向传播的结果。这里不使用滑动平均
    y_halt = inference(x, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量，并将其指定为不可训练的。在使用TensorFlow训练神经网络时，一般将代表轮数的变量指定为不可训练
    global_step = tf.Variable(0, trainable=False)

    # 用滑动平均衰减率和训练轮数变量初始化滑动平均类。给定训练轮数可以加快训练早期变量更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 通过tf.trainable_variables()返回计算图上的可训练变量集合（GraphKeys.TRAINABLE_VARIABLES）中的元素，这个集合中的
    # 元素就是所有trainable=False的变量。对这可训练变量使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果。滑动平均不会改变变量本身的取值，而是维护一个影子变量保存其滑动平均值
    # 当使用滑动平均值时需要明确调用average函数

    average_y_halt = inference(x, weights1, biases1, weights2, biases2, variable_averages)

    # 定以损失函数为交叉熵.
    # tensorflow中提供了sparse_softmax_cross_entropy_with_logits函数计算交叉熵
    # 这个函数的第logits参数是神经网络不包括softmax层的前向传播结果,labels参数是训练数据的标注
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_halt, labels=y)

    # 计算当前batch中所有样本的交叉熵均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失.通常仅计算权值的正则化损失而不计算偏置
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率,学习率在此基础上随着迭代次数而减小
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 一轮迭代所需要的迭代次数
        LEARNIING_RATE_DECAY  # 学习率的衰减率
    )

    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时,每过一遍数据既需要通过反向传播算法来跟新神经网络中的参数,又要更新每一个参数的滑动平均值.
    # tensorflow中提供了tf.cttronl_dependencies和tf.group两种机制实现一次完成多个操作.
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 检查使用了滑动平均模型的神经网络前向传播结果是否正确,并计算准确率.
    correct_prediction = tf.equal(tf.argmax(average_y_halt, 1), tf.argmax(y, 1))
    accurracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 准备验证数据,通常通过验证数据判断训练终止条件和模型训练效果
        validate_feed = {x: mnist.validation.images,
                         y: mnist.validation.labels}

        # 准备测试数据.真实应用中,测试数据是不可见的.测试数据用于评价模型优劣
        test_feed = {x: mnist.test.images,
                     y: mnist.test.labels}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            # 每迭代1000轮输出验证集上的效果
            if i % 1000 == 0:
                # 因为MNIST数据集很小,这里计算验证集上滑动平均模型的预测结果时没有将验证集或分成多个batch.
                # 当神经网络比较复杂或者验证集较大时,太大的batch可能会导致内存溢出
                validate_acc = sess.run(accurracy, feed_dict=validate_feed)
                print("After %d training step(s),validate accuracy using average model is %g" % (i, validate_acc))

            # 产生这一轮使用的一个batch数据,并进行训练
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y: ys})

        # 训练结束之后,在测试集上检测神经网络模型的最终正确率
        test_acc = sess.run(accurracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))


def main(argv=None):
    '''
    主程序入口
    :param argv:
    :return:
    '''
    # 加载MNSIT数据集
    mnist = input_data.read_data_sets("./data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()  # 会自动调用main函数
