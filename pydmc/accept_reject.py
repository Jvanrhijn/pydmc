import math
from abc import ABC, abstractmethod

import numpy as np

from pydmc.node_warp import node_distance
from pydmc.util import velocity_cutoff_umrigar


class AcceptReject(ABC):

    @abstractmethod
    def move_state(self, wave_function, x, time_step, velocity_cutoff):
        pass


class BoxAcceptReject(AcceptReject):

    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)

    def move_state(self, wave_function, x, time_step, velocity_cutoff=lambda v, tau: v):
        value_old = wave_function(x)
        
        xprop = x + time_step/2 * self._rng.uniform(low=-1, high=1, size=x.shape)

        value_new = wave_function(xprop)

        ratio = value_new**2 / value_old**2

        acceptance = min(1, ratio)
        accepted = acceptance > self._rng.uniform()

        return accepted, acceptance, (xprop if accepted else x)


class DiffuseAcceptReject(AcceptReject):

    def __init__(self, seed=0, fixed_node=False):
        self._rng = np.random.default_rng(seed)
        self._fixed_node = fixed_node

    def move_state(self, wave_function, x, time_step): #, velocity_cutoff=lambda v, tau: v):
        # TODO: allow single-electron moves, ignore for now
        #       since our test case only has 1 particle
        value_old = wave_function(x)
        drift_old = velocity_cutoff_umrigar(wave_function.gradient(x) / value_old, time_step)

        xprop = x + drift_old * time_step + self._rng.normal(size=x.shape, scale=math.sqrt(time_step))

        value_new = wave_function(xprop)

        # edge case
        if value_new == 0:
            return False, 0, x

        drift_new = velocity_cutoff_umrigar(wave_function.gradient(xprop) / value_new, time_step)

        # reject if node is crossed and we're doing FN-DMC
        if self._fixed_node and math.copysign(1, value_old) != math.copysign(1, value_new):
            return False, 0, x

        try_num = np.exp(-np.linalg.norm(x - xprop - drift_new*time_step)**2 / (2*time_step))
        try_den = np.exp(-np.linalg.norm(xprop - x - drift_old*time_step)**2 / (2*time_step))

        acceptance = min(1, try_num * value_new**2 / (try_den * value_old**2))
        accepted = acceptance > self._rng.uniform()

        return accepted, acceptance, (xprop if accepted else x)

    def reseed(self, seed):
        self._rng = np.random.default_rng(seed)


class DiffuseAcceptRejectSorella(AcceptReject):

    def __init__(self, seed=0, fixed_node=False, epsilon=1e-2):
        self._rng = np.random.default_rng(seed)
        self._fixed_node = fixed_node
        self._epsilon = epsilon

    def move_state(self, wave_function, x, time_step):
        value_old = wave_function(x)
        grad_old = wave_function.gradient(x)
        drift_old = velocity_cutoff_umrigar(grad_old / value_old, time_step)

        xprop = x + drift_old * time_step + self._rng.normal(size=x.shape, scale=math.sqrt(time_step))

        value_new = wave_function(xprop)

        # edge case
        if value_new == 0:
            return False, 0, x

        grad_new = wave_function.gradient(xprop)
        drift_new = velocity_cutoff_umrigar(grad_new / value_new, time_step)

        # reject if node is crossed and we're doing FN-DMC
        if self._fixed_node and math.copysign(1, value_old) != math.copysign(1, value_new):
            return False, 0, x

        try_num = np.exp(-np.linalg.norm(x - xprop - drift_new*time_step)**2 / (2*time_step))
        try_den = np.exp(-np.linalg.norm(xprop - x - drift_old*time_step)**2 / (2*time_step))

        # compute new distribution according to Sorella
        d = node_distance(grad_old, value_old)
        deps = d if d > self._epsilon \
            else self._epsilon*(d/self._epsilon)**(d/self._epsilon)

        dprime = node_distance(grad_new, value_new)
        deps_prime = dprime if dprime > self._epsilon \
            else self._epsilon*(dprime/self._epsilon)**(dprime/self._epsilon)

        acceptance = min(1, try_num * (value_new * deps_prime / dprime)**2 \
                         / (try_den * (value_old * deps/d)**2))
        accepted = acceptance > self._rng.uniform()

        return accepted, acceptance, (xprop if accepted else x)

    def reseed(self, seed):
        self._rng = np.random.default_rng(seed)


class BoxAcceptRejectSorella(AcceptReject):

    def __init__(self, seed=0, epsilon=1e-2):
        self._rng = np.random.default_rng(seed)
        self._epsilon = epsilon

    def move_state(self, wave_function, x, time_step):
        value_old = wave_function(x)
        grad_old = wave_function.gradient(x)

        xprop = x + time_step/2 * self._rng.uniform(low=-1, high=1, size=x.shape)

        value_new = wave_function(xprop)
        grad_new = wave_function.gradient(xprop)

        # edge case
        if value_new == 0:
            return False, 0, x

        # compute new distribution according to Sorella
        d = node_distance(grad_old, value_old)
        deps = d if d > self._epsilon \
            else self._epsilon*(d/self._epsilon)**(d/self._epsilon)

        dprime = node_distance(grad_new, value_new)
        deps_prime = dprime if dprime > self._epsilon \
            else self._epsilon*(dprime/self._epsilon)**(dprime/self._epsilon)

        acceptance = min(1, (value_new * deps_prime / dprime)**2 \
                         / (value_old * deps/d)**2)
        accepted = acceptance > self._rng.uniform()

        return accepted, acceptance, (xprop if accepted else x)

    def reseed(self, seed):
        self._rng = np.random.default_rng(seed)