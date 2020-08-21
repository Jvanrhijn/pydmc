import copy
import math
import multiprocessing

import numpy as np
import tqdm

from pydmc.accept_reject import *
from pydmc.branching import *
from pydmc.walker import *
from pydmc.wavefunction import *
from pydmc.util import chunks


class DMC:

    def __init__(self, hamiltonian, walkers, brancher, ar, guiding_wf, reference_energy, force_accumulator=None, seed=1):
        self._hamiltonian = hamiltonian
        self._walkers = walkers
        self._brancher = brancher
        self._ar = ar
        self._guiding_wf = guiding_wf
        self._reference_energy = reference_energy
        self._energy_all = []
        self._energy_cumulative = [reference_energy]
        self._variance = [0.0]
        self._confs = []
        self._error = [0.0]
        self.force_accumulator = force_accumulator

    def run_dmc(self, time_step, num_blocks, steps_per_block, accumulator=None, neq=1, progress=True):

        if progress:
            range_wrapper = tqdm.tqdm
        else:
            range_wrapper = lambda x: x

        for b in range_wrapper(range(num_blocks)):

            block_energies = np.zeros(steps_per_block)

            for i in range(steps_per_block):

                ensemble_energy = 0
                total_weight = 0

                for iwalker, walker in enumerate(self._walkers):
                    self._confs.append(walker.configuration)
                    local_energy = self._update_walker(walker, time_step)
                    ensemble_energy += walker.weight*local_energy
                    total_weight += walker.weight

                    if self.force_accumulator is not None and b >= neq:
                        self.force_accumulator.accumulate_samples(
                            iwalker,
                            walker, 
                            self._guiding_wf, 
                            self._hamiltonian, 
                            self._reference_energy, 
                            time_step
                        )
                
                    if accumulator is not None:
                        accumulator.sample_observables(self._guiding_wf, walker)
                        accumulator.ref_energy.append(walker.weight*self._reference_energy)

                ensemble_energy /= total_weight
                block_energies[i] = ensemble_energy

                self._energy_all.append(ensemble_energy)

                self._brancher.perform_branching(self._walkers)


            # skip equilibration blocks
            if b >= neq:
                block_average_energy = np.mean(block_energies)
                self.update_energy_estimate(block_average_energy)
                self._reference_energy = (self._reference_energy + self._energy_cumulative[-1]) / 2

        return accumulator

    def _update_walker(self, walker, time_step):
        xold = copy.deepcopy(walker.configuration)
        wf_value_old = self._guiding_wf(xold)

        # compute "old" local energy
        local_energy_old = self._hamiltonian(self._guiding_wf, xold) / wf_value_old

        # perform accept/reject step
        accepted, acceptance_prob, xnew = self._ar.move_state(self._guiding_wf, xold, time_step)

        # compute "new" local energy
        wf_value_new = self._guiding_wf(xnew)
        local_energy_new = self._hamiltonian(self._guiding_wf, xnew) / wf_value_new

        # update walker weight and configuration
        s = self._reference_energy - local_energy_old
        sprime = self._reference_energy - local_energy_new
        walker.weight *= math.exp((0.5*acceptance_prob*(s + sprime) + (1- acceptance_prob)*s)*time_step)
        walker.configuration = xnew

        return local_energy_new

    @property
    def energy_estimate(self):
        return np.array(self._energy_cumulative)

    @property
    def energy_error(self):
        return np.array(self._error)

    def update_energy_estimate(self, energy_new):
        idx = len(self._energy_cumulative) - 1
        energy_prev = self._energy_cumulative[-1]
        self._energy_cumulative.append(energy_prev + (energy_new - energy_prev) / (idx + 2))
        self._variance.append(self._variance[-1] + ((energy_new - energy_prev)*(energy_new - self._energy_cumulative[-1]))/(idx + 1))
        self._error.append(math.sqrt(self._variance[-1] / (idx + 1)))


class Accumulator:

    def __init__(self, observables):
        self.observables = observables
        self.observables_data = {name: [] for name in observables}
        self.ref_energy = []
        self.weights = []

    def sample_observables(self, guiding_wf, walker):
        self.weights.append(walker.weight)
        for name, function in self.observables.items():
            sample = function(guiding_wf, walker.configuration)
            self.observables_data[name].append(sample)