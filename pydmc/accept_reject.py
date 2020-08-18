import math
from abc import ABC, abstractmethod

import numpy as np


class AcceptReject(ABC):

    @abstractmethod
    def move_state(self, wave_function, x):
        pass


class DiffuseAcceptReject(AcceptReject):

    def __init__(self, time_step, seed=0, fixed_node=False):
        self._rng = np.random.default_rng(seed)
        self._time_step = time_step
        self._fixed_node = fixed_node

    def _propose(self, wave_function, x):
        # TODO: allow single-electron moves, ignore for now
        #       since our test case only has 1 particle
        value_old = wave_function(x)
        gradient_old = wave_function.gradient(x)
        drift_old = gradient_old / value_old
        return x + drift_old * self._time_step + math.sqrt(self._time_step)*self._rng.normal()

    def _accept(self, wave_function, x, xprop):
        value_old = wave_function(x)
        value_new = wave_function(xprop)
        drift_old = wave_function.gradient(x) / value_old
        drift_new = wave_function.gradient(xprop) / value_new
        # reject if node is crossed and we're doing FN-DMC
        if self._fixed_node and math.copysign(1, value_old) != math.copysign(1, value_new):
            return x
        
        try_num = np.exp(-np.linalg.norm(x - xprop - drift_new*self._time_step)**2 / (2*self._time_step))
        try_den = np.exp(-np.linalg.norm(xprop - x - drift_old*self._time_step)**2 / (2*self._time_step))

        acceptance = try_num * value_new**2 / (try_den * value_old**2)

        return xprop if acceptance > self._rng.uniform() else x