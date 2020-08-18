import numpy


class Walker:

    def __init__(self, configuration, weight):
        self._configuration = configuration
        self._weight = weight

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
        self._configuration = conf