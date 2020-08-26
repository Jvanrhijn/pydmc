import numpy


class Walker:

    def __init__(self, configuration, weight):
        self._configuration = configuration
        self._weight = weight
        self._configuration_prev = None
        self._branching_factor = None

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