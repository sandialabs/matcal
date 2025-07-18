from abc import ABC, abstractmethod
import numpy as np

class ClusterBase(ABC):

    @abstractmethod
    def fit_predict(self):
        pass

    @abstractmethod
    def predict(self):
        pass

class MedianSplit(ClusterBase):

    def __init__(self, split_axis:int):
        self._split_axis = split_axis
        self._median = None

    def fit_predict(self, points):
        axis_points = self._get_axis_points(points)
        self._median = np.median(axis_points)
        cluster = self._predict_from_axis(axis_points)
        return cluster

    def _predict_from_axis(self, axis_points):
        cluster = axis_points >= self._median
        return cluster

    def _get_axis_points(self, points):
        axis_points = points[:, self._split_axis]
        return axis_points

    def predict(self, points):
        axis_points = self._get_axis_points(points)
        cluster = self._predict_from_axis(axis_points)
        return cluster