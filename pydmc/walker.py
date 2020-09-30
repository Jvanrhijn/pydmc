import numpy
from collections import deque


class Walker:

    def __init__(self, configuration, weight, lag=1):
        self._configuration = configuration
        self._weight = weight
        self._configuration_prev = None
        self._branching_factor = None
        self._value = None
        self._gradient = None
        self._previous_value = None
        self._previous_gradient = None
        self.histories = {
                "grad_a S": deque(maxlen=lag),
                "grad_a T": deque(maxlen=lag),
                "grad_a S (warp)": deque(maxlen=lag),
                "grad_a T (warp)": deque(maxlen=lag),
                "grad_a S (no cutoff)": deque(maxlen=lag),
                "grad_a T (no cutoff)": deque(maxlen=lag),
                "grad_a S (warp, no cutoff)": deque(maxlen=lag),
                "grad_a T (warp, no cutoff)": deque(maxlen=lag),
                "grad_a sum Log Jacobian": deque(maxlen=lag),
        }

    @property
    def value(self):
        return self._value

    @property
    def previous_value(self):
        return self._value_old

    @value.setter
    def value(self, new):
        self._value_old = self._value
        self._value = new

    @property
    def gradient(self):
        return self._gradient

    @property
    def previous_gradient(self):
        return self._gradient_old

    @gradient.setter
    def gradient(self, new):
        self._gradient_old = self._gradient
        self._gradient = new

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, w):
        self._weight = w

    @property
    def configuration(self):
        return self._configuration

    @configuration.setter
    def configuration(self, conf):
        self._configuration_prev = self._configuration
        self._configuration = conf

    @property
    def previous_configuration(self):
        return self._configuration_prev

    @property
    def branching_factor(self):
        return self._branching_factor

    @branching_factor.setter
    def branching_factor(self, bf):
        self._branching_factor = bf

    def __repr__(self):
        return f"W: {self.weight}, C: {self.configuration}"