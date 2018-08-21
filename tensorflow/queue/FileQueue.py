#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-20 下午9:03
# @Author  : yinwb
# @File    : FileQueue.py
'''
模拟海量数据情况下将数据写入不同的文件。并使用文件列表读取数据。
'''
import tensorflow as tf

if __name__ == "__main__":
    # 数据分区数量，每个分区保存为一个文件
    PARTITION_NUM = 2
    # 每个分区的样本数量
    INSTANCE_PER_PARTITION = 3

    for i in range(PARTITION_NUM):
        filename = "./data/data.tfrecords-%.5d-of-%.5d" % (i, PARTITION_NUM)
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(INSTANCE_PER_PARTITION):
            example = tf.train.Example(features=tf.train.Features(feature={
                "i": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                "j": tf.train.Feature(int64_list=tf.train.Int64List(value=[j]))
            }))
            writer.write(example.SerializeToString())
        writer.close()

    # 使用tf.train.match_filenames_once获取文件列表
    files = tf.train.match_filenames_once("./data/data.tfrecords-*")

    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "i": tf.FixedLenFeature([], tf.int64),
            "j": tf.FixedLenFeature([], tf.int64)
        }
    )

    with tf.Session() as sess:
        # 使用tf.train.match_filenames_once函数时需要初始化一些变量，因此需要显示调用变量初始化操作
        sess.run(tf.local_variables_initializer())
        print(sess.run(files))
        # 声明tf.train.Coordinator类来协同不同线程，并启动线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 多次执行获取数据操作
        for i in range(6):
            print(sess.run([features["i"], features["j"]]))

        coord.request_stop()
        coord.join(threads)
