from abc import ABC, abstractmethod
from functools import reduce

import numpy as np

from walker import Walker


class Brancher(ABC):

    @abstractmethod
    def perform_branching(self, walkers):
        pass


class SRBrancher(Brancher):

    def perform_branching(self, walkers):
        weights = np.array([walker.weight for walker in walkers])
        global_weight = np.mean(weights)
        new_walkers = []
        for _ in range(len(walkers)):
            new_walkers.append(Walker(np.random.choice(walkers, p=weights).configuration, global_weight))
        return new_walkers