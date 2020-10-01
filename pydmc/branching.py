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
        global_weight = np.mean(weights)
        weights /= global_weight
        new_walkers = random.choices(walkers, weights=weights, k=len(walkers))

        # random.choices returns references to original objects, which
        # reduces the population size. Copy each walker to prevent this.
        new_walkers = [copy.deepcopy(walker) for walker in new_walkers]

        for walker in new_walkers:
            walker.weight = 1.0 #global_weight
        
        return new_walkers


class OptimalSRBrancher(Brancher):

    def perform_branching(self, walkers):
        weights = np.array([walker.weight for walker in walkers])
        global_weight = np.mean(weights)

        positive_walkers = list(filter(lambda w: w.weight/global_weight > 1, walkers))
        negative_walkers = list(filter(lambda w: w.weight/global_weight < 1, walkers))

        num_reconf = int(sum(abs(w.weight / global_weight - 1) for w in positive_walkers) + np.random.uniform())
        num_reconf = min(num_reconf, len(negative_walkers), len(positive_walkers))

        pos_weights = np.array(list(w.weight for w in positive_walkers))
        pos_weights /= pos_weights.sum()
        neg_weights = np.array(list(w.weight for w in negative_walkers))
        neg_weights /= neg_weights.sum()

        if num_reconf > 0:

            # acquire a random set of walkers to duplicate and one to destroy
            to_duplicate = np.random.choice(list(range(len(positive_walkers))), size=num_reconf, p=None, replace=False)
            to_destroy = np.random.choice(list(range(len(negative_walkers))), size=num_reconf, p=None, replace=False)

            # destroy the chosen negative walkers
            for i in sorted(to_destroy)[::-1]:
                del negative_walkers[i]

            # duplicate the chosen positive walkers
            for i in to_duplicate:
                positive_walkers.append(copy.deepcopy(positive_walkers[i]))

        new_walkers = negative_walkers + positive_walkers

        # reset walker weights to unity
        for walker in new_walkers:
            walker.weight = global_weight

        return new_walkers


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
            #new_walkers.append(Walker(walker.configuration, new_weight))
            new_walker = copy.deepcopy(walker)
            new_walker.weight = new_weight
            new_walkers.append(new_walker)
        
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
            new_walker = copy.deepcopy(w)
            new_walker.weight = new_weight
            new_walkers.append(new_walker)
            #new_walkers.append(Walker(w.configuration, new_weight))

        return new_walkers


class NoBrancher(Brancher):

    def perform_branching(self, walkers):
        return walkers