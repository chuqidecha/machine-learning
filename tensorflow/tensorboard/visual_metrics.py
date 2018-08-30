#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-30 下午11:19
# @Author  : yinwb
# @File    : visual_metrics.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "./log/visual_metrics"

BATCH_SIZE = 128

EPOCH = 10


def variable_summaries(var, name):
    with tf.variable_scope("log-summaries"):
        tf.summary.histogram(name, var)

        mean = tf.reduce_mean(var)

        tf.summary.scalar("mean/" + name, mean)

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar("stddev/" + name, stddev)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.variable_scope(layer_name):
        with tf.variable_scope("weights"):
            weights = tf.get_variable("weights", shape=[input_dim, output_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            variable_summaries(weights, layer_name + "/weights")

        with tf.variable_scope("biases"):
            biases = tf.get_variable("biases", shape=[output_dim], initializer=tf.constant_initializer(0.0))
            variable_summaries(weights, layer_name + "/biases")

        with tf.variable_scope("xw_plus_b"):
            xw_plus_b = tf.nn.xw_plus_b(input_tensor, weights, biases)
            tf.summary.histogram(layer_name + "/xw_plus_b", xw_plus_b)

        activation = act(xw_plus_b, name="activation")
        tf.summary.histogram(layer_name + "/activation", activation)
        return activation


def main(_):
    mnist = input_data.read_data_sets("../mnist/data", one_hot=True)
    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784], name="x-input")
        y = tf.placeholder(tf.float32, [None, 10], name="y-input")

    with tf.variable_scope("input-reshape"):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image("input-reshape", image_shaped_input, 10)

    hidden1 = nn_layer(x, 784, 500, "layer1-fc")
    y_halt = nn_layer(hidden1, 500, 10, "layer2-fc", act=tf.identity)

    with tf.variable_scope("cross-entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_halt, labels=y))
        tf.summary.scalar("cross entropy", cross_entropy)

    with tf.variable_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    with tf.variable_scope("accuracy"):
        with tf.variable_scope("correct-prediction"):
            correct_prediction = tf.equal(tf.argmax(y_halt, 1), tf.argmax(y, 1))
        with tf.variable_scope("accuracy"):
            acurracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", acurracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

        sess.run(tf.global_variables_initializer())

        steps = mnist.train.num_examples // BATCH_SIZE * EPOCH
        for i in range(steps):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y: ys})

            summary_writer.add_summary(summary, i)


if __name__ == '__main__':
    tf.app.run()
