#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-19 下午9:23
# @Author  : yinwb
# @File    : Queue.py

import tensorflow as tf

if __name__ == "__main__":
    # 创建一个先进先出队列,指定队列中最多可以保存两个元素,并指定类型为整数
    q = tf.FIFOQueue(2, "int32")

    # 使用enqueue_many函数来初始化队列中的元素.使用队列之前要明确初始化队列过程.
    init = q.enqueue_many(([0, 10],))
    # 使用dequeue函数将队列中第一个元素出队列.
    x = q.dequeue()

    # 将得到的值加1后重新加入队列
    y = x + 1
    q_inc = q.enqueue([y])

    with tf.Session() as sess:
        # 初始化队列操作
        init.run()
        for _ in range(5):
            v, _ = sess.run([x, q_inc])
            print(v)
