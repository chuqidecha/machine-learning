#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-7-17 下午9:05
# @Author  : yinwb
# @File    : decisionTree.py

import numpy as np
import pickle

from inaction.classification import Classification
from inaction.decisiontree.tools import *



class DecisionTree(Classification):
    def __init__(self, modelType="ID3", colNames=None, alpha=None):
        self.setModelType(modelType)
        self.__colNames = colNames
        self.__alpha = alpha
        self.__tree = None

    def getModelType(self):
        return self.__modelType

    def setModelType(self, modelType):
        if modelType not in ["ID3", "C4.5"]:
            raise ValueError("Unknown Decision Tree Model")
        self.__modelType = modelType
        return self

    def getColNames(self):
        return self.__colNames

    def setColNames(self, colNames):
        self.__colNames = colNames
        return self

    def getAlpha(self):
        return self.__alpha

    def setAlpha(self, alpha):
        self.__alpha = alpha
        return self

    def getTree(self):
        return self.__tree

    def __selectFeature(self, dataSet):
        '''
        特征选择函数
        :param dataSet: 样本 ndarray m*n
        :return: 特征索引
        '''
        numFeat = dataSet.shape[1] - 1
        maxInfoGain = 0.0
        bestFeat = -1
        # 特征选择，如果是ID3则选择信息增益最大的特征，如果是C4.5则选择信息增益比最大的特征
        for feat in range(numFeat):
            infoGain = 0.0
            if self.__modelType == "ID3":
                infoGain = calcInfoGain(dataSet[:, -1].flatten(), dataSet[:, feat].flatten())
            else:
                infoGain = calcInfoGainRatio(dataSet[:, -1].flatten(), dataSet[:, feat].flatten())
            if infoGain > maxInfoGain:
                maxInfoGain = infoGain
                bestFeat = feat
        return bestFeat

    def __buildTree(self, dataSet, featNames):
        '''
        递归创建决策树
        :param dataSet: 样本 ndarray m*n
        :param featNames: 样本各列的名称
        :return: 嵌套字典表示的决策树
        '''
        uniqueVal = np.unique(dataSet[:, -1])
        # 所有样本属于同一个类别
        if uniqueVal.shape[0] == 1:
            return uniqueVal[0]

        # 所有特征都已使用，采用投票法
        if featNames.shape[0] == 0:
            return vote(dataSet[:, -1])

        # 选择最佳分割特征
        bestFeat = self.__selectFeature(dataSet)
        bestFeatName = featNames[bestFeat]

        featIndexes = (np.arange(dataSet.shape[1]) != bestFeat)
        subFeatNames = featNames[featIndexes]
        tree = {bestFeatName: {}}

        # 按属性值分割样本，递归创建
        featValueSet = np.unique(dataSet[:, bestFeat])
        for featValue in featValueSet:
            subDataSet = dataSet[dataSet[:, bestFeat] == featValue, :][:, featIndexes]
            tree[bestFeatName][featValue] = self.__buildTree(subDataSet, subFeatNames)
        return tree

    def __pruning(self, tree, dataSet):
        '''
        剪枝
        :param tree:
        :param dataSet:
        :return: （子树、叶子节点数、不确定度）
        '''
        # 不剪枝时的损失
        cost = dataSet.shape[0] * calcEntropy(dataSet[:, -1]) + self.__alpha
        leaves = 1

        # 如果包含子树，递归剪枝,
        if isinstance(tree, dict):
            root = list(tree.keys())[0]
            featIndex = np.argwhere(self.__colNames == root)[0, 0]

            # 剪枝时的损失
            subRes = []
            for key in tree[root].keys():
                res = self.__pruning(tree[root][key], dataSet[dataSet[:, featIndex] == key])
                subRes.append(res)
            leaves = sum([res[1] for res in subRes])
            cost1 = sum([res[2] for res in subRes])

            # 剪枝
            if cost < cost1:
                tree = vote(dataSet[:, -1])
                leaves = 1
            else:
                cost = cost1
        return tree, 1, cost

    def fit(self, dataSet):
        '''
        训练决策树
        :param dataSet: 样本 ndarray m*n
        :return: 对象本身
        '''
        if  self.__colNames is None:
            self.__colNames = np.arange(dataSet.shape[1])
        self.__tree = self.__buildTree(dataSet, self.__colNames)
        if self.__alpha is not None:
            self.__pruning(self.__tree, dataSet)
        return self

    # TODO 所有参数用dict序列化
    def load(self, fileName):
        '''
        从外部文件加载决策树
        :param fileName: 文件名
        :return:
        '''
        with open(fileName, 'rb') as fr:
            self.__tree = pickle.load(fr)
            return self

    def transform(self, dataSet):
        for sample in dataSet:
            tmp = self.__tree
            while isinstance(tmp,dict):
                node = list(tmp.keys())[0]
                featIndex = np.argwhere(self.__colNames == node)[0, 0]
                tmp = tmp[node][sample[featIndex]]
            return tmp

    def save(self, fileName):
        with open(fileName, 'wb') as fw:
            pickle.dump(self.__tree, fw)

    def __str__(self):
        import json
        return json.dumps(self.__tree, ensure_ascii=False)


if __name__ == '__main__':
    dataSet = np.array([['青年', '否', '否', '一般', '否'],
                        ['青年', '否', '否', '好', '否'],
                        ['青年', '是', '否', '好', '是'],
                        ['青年', '是', '是', '一般', '是'],
                        ['青年', '否', '否', '一般', '否'],
                        ['中年', '否', '否', '一般', '否'],
                        ['中年', '否', '否', '好', '否'],
                        ['中年', '是', '是', '好', '是'],
                        ['中年', '否', '是', '非常好', '是'],
                        ['中年', '否', '是', '非常好', '是'],
                        ['老年', '否', '是', '非常好', '是'],
                        ['老年', '否', '是', '好', '是'],
                        ['老年', '是', '否', '好', '是'],
                        ['老年', '是', '否', '非常好', '是'],
                        ['老年', '否', '否', '一般', '否']])
    colNames = np.array(['年龄', '有工作', "有自己的房子", "信贷情况", "类别"])

    id3 = DecisionTree().setModelType("ID3").setAlpha(1).setColNames(colNames).setAlpha(2)

    print(id3.fit(dataSet))
    print(id3.transform([['老年', '否', '否', '一般']]))
    print(id3.transform([['中年', '否', '是', '非常好']]))
    TreePlot().plot(id3.getTree())
