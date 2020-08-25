import random
import math
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
        confs = [walker.configuration for walker in walkers]
        new_confs = random.choices(confs, weights=weights, k=len(walkers))
        return [Walker(conf, global_weight) for conf in new_confs]


class SimpleBrancher(Brancher):

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


class SplitJoinBrancher(Brancher):

    def perform_branching(self, walkers):
        new_walkers = []

        to_split = []
        to_join = []

        for walker in walkers:
            if walker.weight > 2:
                to_split.append(walker)
            elif walker.weight < 0.5:
                to_join.append(walker)
            else:
                new_walkers.append(walker)

        # perform splitting
        for walker in to_split:
            num_new_walkers = math.floor(walker.weight)
            new_weight = walker.weight / num_new_walkers
            new_walkers.append(Walker(walker.configuration, new_weight))
        
        # perform joining
        for i in range(0, len(to_split), 2):

            chunk = to_join[i:i+2]

            if len(chunk) < 2:
                continue

            w1, w2 = to_join[i:i+2]
            new_weight = w1.weight + w2.weight

            if new_weight == 0:
                continue

            p1 = w1.weight / new_weight
            p2 = 1 - p1
            w = np.random.choice(chunk, p=[p1, p2])
            new_walkers.append(Walker(w.configuration, new_weight))

        return new_walkers


class NoBrancher(Brancher):

    def perform_branching(self, walkers):
        return walkers