import copy
from abc import ABC, abstractmethod
from functools import reduce

import numpy as np

from pydmc.walker import Walker


class Brancher(ABC):

    @abstractmethod
    def perform_branching(self, walkers):
        pass


class SRBrancher(Brancher):

    def perform_branching(self, walkers):
        weights = np.array([walker.weight for walker in walkers])
        weights /= weights.sum()
        global_weight = np.mean(weights)
        new_walkers = []
        for _ in range(len(walkers)):
            new_walkers.append(Walker(np.random.choice(walkers, p=weights).configuration, global_weight))
        return new_walkers


class SimpleBrancher(Brancher):

    def __init__(self, max_copies=50):
        self._max_copies = max_copies

    def perform_branching(self, walkers):
        extra_walkers = []
        for i, walker in enumerate(walkers):
            if walker.weight > 1:
                num_copies = min(3, int(walker.weight + np.random.uniform()) - 1)
                for _ in range(num_copies):
                    extra_walkers.append(copy.deepcopy(walker))
            else:
                del walkers[i]
        return walkers + extra_walkers


            