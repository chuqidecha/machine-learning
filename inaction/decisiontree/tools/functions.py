#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-7-17 下午9:05
# @Author  : yinwb
# @File    : __init__.py

import numpy as np
from collections import Counter


def calcEntropy(X):
    '''
    计算数据集的熵
    :param X: 列向量,numpy.array
    :return: 熵
    '''

    # 统计各个类别的样本数
    _, counts = np.unique(X, return_counts=True)
    # 计算数据集的熵
    entropy = -np.dot(np.true_divide(counts, np.sum(counts)), np.log2(np.true_divide(counts, np.sum(counts))))
    return entropy


def calcInfoGain(D, A):
    '''
    信息增益
    :param D 数据集
    :param X D|X
    :return:
    '''
    baseEntropy = calcEntropy(D)
    uniqueVal, counts = np.unique(A, return_counts=True)

    subSetEntropy = []
    for val in uniqueVal:
        subSet = D[A == val]
        subSetEntropy.append(calcEntropy(subSet))
    conditionEntropy = np.sum(np.dot(np.true_divide(counts, np.sum(counts)), np.array(subSetEntropy)))
    return baseEntropy - conditionEntropy


def calcInfoGainRatio(D, A):
    '''
    计算信息增益比
    :param dataSet:
    :return:
    '''
    entropyGain = calcInfoGain(D, A)
    entropy = calcEntropy(A)
    return entropyGain / entropy


def calcGini(D):
    # 统计各个类别的样本数
    _, counts = np.unique(D, return_counts=True)
    # 计算数据集的Gini不纯度
    giniImpurity = 1 - np.sum(np.power(np.true_divide(counts, np.sum(counts)), 2))
    return giniImpurity

def calcRSS(y,yHalt):
    return np.sum(np.power(yHalt - y, 2))


def vote(array):
    return Counter(array).most_common(1)[0][0]
