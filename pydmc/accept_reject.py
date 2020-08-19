import math
from abc import ABC, abstractmethod

import numpy as np

from pydmc.util import velocity_cutoff


class AcceptReject(ABC):

    @abstractmethod
    def move_state(self, wave_function, x, time_step):
        pass


class DiffuseAcceptReject(AcceptReject):

    def __init__(self, seed=0, fixed_node=False):
        self._rng = np.random.default_rng(seed)
        self._fixed_node = fixed_node

    def move_state(self, wave_function, x, time_step):
        # TODO: allow single-electron moves, ignore for now
        #       since our test case only has 1 particle
        value_old = wave_function(x)
        drift_old = velocity_cutoff(wave_function.gradient(x) / value_old, time_step)

        xprop = x + drift_old * time_step + self._rng.normal(size=x.shape, scale=math.sqrt(time_step))

        value_new = wave_function(xprop)
        drift_new = velocity_cutoff(wave_function.gradient(xprop) / value_new, time_step)

        # reject if node is crossed and we're doing FN-DMC
        if self._fixed_node and math.copysign(1, value_old) != math.copysign(1, value_new):
            return False, 0, x

        try_num = np.exp(-np.linalg.norm(x - xprop - drift_new*time_step)**2 / (2*time_step))
        try_den = np.exp(-np.linalg.norm(xprop - x - drift_old*time_step)**2 / (2*time_step))

        acceptance = min(1, try_num * value_new**2 / (try_den * value_old**2))
        accepted = acceptance > self._rng.uniform()

        return accepted, acceptance, (xprop if accepted else x)