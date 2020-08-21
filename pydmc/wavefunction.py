from abc import ABC, abstractmethod

import numpy as np


class WaveFunction(ABC):

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @abstractmethod
    def laplacian(self, x):
        pass


class Parametrized(ABC):

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def parameter_gradient(self):
        pass

    @abstractmethod
    def set_parameters(self, ps):
        pass


class GeometryParametrized(ABC):

    @abstractmethod
    def geometry_parameters(self):
        pass

    @abstractmethod
    def deform(self, parameter_change):
        pass