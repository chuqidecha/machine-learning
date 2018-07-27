# -*- coding: UTF-8 -*-
import abc


class Classification(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, dataSet):
        pass

    @abc.abstractmethod
    def transform(self, dataSet):
        pass

    @abc.abstractmethod
    def load(self, fileName):
        pass

    @abc.abstractmethod
    def save(self):
        pass
