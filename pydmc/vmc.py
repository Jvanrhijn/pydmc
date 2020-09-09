import copy
import math

import numpy as np
import tqdm

from pydmc.accept_reject import *
from pydmc.branching import *
from pydmc.walker import *
from pydmc.wavefunction import *
from pydmc.util import chunks


class VMC:

    def __init__(self, hamiltonian, initial_conf, ar, guiding_wf, force_accumulators=None, seed=1, velocity_cutoff=lambda v, tau: v):
        self._hamiltonian = hamiltonian
        self._conf = initial_conf
        self._ar = ar
        self._guiding_wf = guiding_wf
        self._energy_all = []
        self._energy_cumulative = []
        self._variance = []
        self._error = []
        self.force_accumulators = force_accumulators
        self._velocity_cutoff = velocity_cutoff

    def run_vmc(self, time_step, num_blocks, steps_per_block, accumulator=None, neq=1, progress=True):

        if progress:
            range_wrapper = tqdm.tqdm
        else:
            range_wrapper = lambda x: x

        for b in range_wrapper(range(num_blocks)):

            block_energies = np.zeros(steps_per_block)

            for i in range(steps_per_block):

                local_energy = self._update(time_step)

                if self.force_accumulators is not None and b >= neq:
                    for fa in self.force_accumulators:
                        fa.accumulate_samples(
                            self._conf, 
                            self._guiding_wf, 
                            self._hamiltonian, 
                            self._energy_cumulative[-1],
                            time_step,
                            self._velocity_cutoff
                        )
                
                block_energies[i] = local_energy

                self._energy_all.append(local_energy)

            # skip equilibration blocks
            if b >= neq:
                block_average_energy = np.mean(block_energies)
                self.update_energy_estimate(block_average_energy)

        return accumulator

    def _update(self, time_step):
        xold = copy.deepcopy(self._conf)

        # perform accept/reject step
        accepted, acceptance_prob, xnew = self._ar.move_state(self._guiding_wf, xold, time_step, self._velocity_cutoff)

        # compute "new" local energy
        wf_value_new = self._guiding_wf(xnew)
        local_energy_new = self._hamiltonian(self._guiding_wf, xnew) / wf_value_new

        self._conf = xnew

        return local_energy_new

    @property
    def energy_estimate(self):
        return np.array(self._energy_cumulative)

    @property
    def energy_error(self):
        return np.array(self._error)

    def update_energy_estimate(self, energy_new):
        if not self._energy_cumulative:
            self._energy_cumulative.append(energy_new)
            self._variance.append(0)
            self._error.append(0)
            return None

        idx = len(self._energy_cumulative) - 1
        energy_prev = self._energy_cumulative[-1]
        self._energy_cumulative.append(energy_prev + (energy_new - energy_prev) / (idx + 2))
        self._variance.append(self._variance[-1] + ((energy_new - energy_prev)*(energy_new - self._energy_cumulative[-1]))/(idx + 1))
        self._error.append(math.sqrt(self._variance[-1] / (idx + 1)))
