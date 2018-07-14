#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-7-17 下午9:05
# @Author  : yinwb
# @File    : treeplot.py

import matplotlib.pyplot as plt


class TreePlot:
    def __init__(self):
        self.__decisionNode = dict(boxstyle="sawtooth", fc="0.8")
        self.__leafNode = dict(boxstyle="round4", fc="0.8")
        self.__arrowArgs = dict(arrowstyle="<-")
        self.__ax1 = None
        self.__xOff = 0
        self.__yOff = 0

    def __getNumberLeaves(self, tree):
        '''
        获取树的叶子结点数
        :param tree:
        :return:
        '''
        leaves = 0
        if type(tree).__name__ == 'dict':
            for key in tree:
                leaves += self.__getNumberLeaves(tree[key])
        else:
            leaves += 1
        return leaves

    def __getTreeDepth(self, tree):
        '''
        获取树的高度
        :param tree:
        :return:
        '''
        depth = 0
        root = list(tree.keys())[0]
        childTree = tree[root]
        for key in childTree.keys():
            if type(childTree[key]).__name__ == 'dict':
                childTreeDepth = self.__getTreeDepth(childTree[key]) + 1
            else:
                childTreeDepth = 1
            if childTreeDepth > depth:
                depth = childTreeDepth
        return depth

    def __plotMidText(self, cntrPr, parentPr, text):
        '''
         父结点和子结点之间的文本注释
        :param cntrPr: 当前子结点坐标
        :param parentPr: 父结点坐标
        :param text: 文本注释
        :return:
        '''
        xMid = (parentPr[0] - cntrPr[0]) / 2 + cntrPr[0]
        yMid = (parentPr[1] - cntrPr[1]) / 2 + cntrPr[1]
        self.__ax1.text(xMid, yMid, text, va="center", ha="center", rotation=30)

    def __plotNode(self, nodeText, centerPt, parentPt, nodeType):
        '''
        绘制树结点
        :param nodeText: 结点注释
        :param centerPt: 当前结点坐标
        :param parentPt: 父结点坐标
        :param nodeType: 结点类型
        :return:
        '''
        self.__ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=self.__arrowArgs)

    def plot(self, tree):
        '''
         绘制树
        :param tree:
        :return:
        '''
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.__ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
        totalLeaves = self.__getNumberLeaves(tree)
        totalDepth = self.__getTreeDepth(tree)
        self.__xOff = -0.5 / totalLeaves
        self.__yOff = 1.0
        self.__plotTree(tree, (0.5, 1.0), '', totalDepth, totalLeaves)
        plt.show()

    def __plotTree(self, tree, parentPt, nodeText, totalDepth, totalLeaves):
        '''
        绘制树
        :param tree: 树
        :param parentPt: 父结点坐标
        :param nodeText: 结点注释
        :param totalDepth: 树的高度
        :param totalLeaves: 叶子结点数
        :return:
        '''
        leaves = self.__getNumberLeaves(tree)
        depth = self.__getTreeDepth(tree)
        root = list(tree.keys())[0]
        cntrPt = (self.__xOff + (1.0 + leaves) / 2.0 / totalLeaves, self.__yOff)
        self.__plotMidText(cntrPt, parentPt, nodeText)
        self.__plotNode(root, cntrPt, parentPt, self.__decisionNode)
        childTree = tree[root]
        self.__yOff = self.__yOff - 1.0 / totalDepth
        for key in childTree.keys():
            # 非叶子结点
            if type(childTree[key]).__name__ == 'dict':
                self.__plotTree(childTree[key], cntrPt, str(key), totalDepth, totalLeaves)
            # 叶子结点
            else:
                self.__xOff = self.__xOff + 1.0 / totalLeaves
                self.__plotNode(childTree[key], (self.__xOff, self.__yOff), cntrPt, self.__leafNode)
                self.__plotMidText((self.__xOff, self.__yOff), cntrPt, str(key))
        self.__yOff = self.__yOff + 1.0 / totalDepth
