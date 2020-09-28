import copy
import math
import multiprocessing
from datetime import datetime

import numpy as np
import tqdm

from pydmc.accept_reject import *
from pydmc.branching import *
from pydmc.walker import *
from pydmc.wavefunction import *
from pydmc.util import chunks


class DMC:

    def __init__(self, hamiltonian, walkers, brancher, ar, guiding_wf, reference_energy, force_accumulators=None, seed=1, velocity_cutoff=lambda v, tau: v):
        self._hamiltonian = hamiltonian

        self._walkers = walkers

        self._brancher = brancher
        self._ar = ar
        self._guiding_wf = guiding_wf
        self._initialize_walkers()

        self._reference_energy = reference_energy
        self._energy_all = []
        self._energy_cumulative = [reference_energy]
        self._variance = [0.0]
        self._confs = []
        self._error = [0.0]

        self.force_accumulators = force_accumulators
        self._velocity_cutoff = velocity_cutoff

    def _initialize_walkers(self):
        for walker in self._walkers:
            x = walker.configuration
            walker.value = self._guiding_wf(x)
            walker.gradient = self._guiding_wf.gradient(x)

    def run_dmc(self, time_step, num_blocks, steps_per_block, neq=1, progress=True, verbose=False):
        start_time = datetime.now()

        if progress and verbose:
            raise ValueError("Can't output a progress bar and logging data")

        if progress:
            range_wrapper = tqdm.tqdm
        else:
            range_wrapper = lambda x: x

        for b in range_wrapper(range(num_blocks)):
            
            block_energies = self._run_block(steps_per_block, b, neq, time_step)

            block_average_energy = np.mean(block_energies)

            if verbose:
                outstring = f"Time elapsed: {datetime.now() - start_time}"
                outstring += f" | Block: {str(b+1).zfill(1+math.ceil(math.log10(num_blocks)))}/{num_blocks} | "
                outstring += f"Energy estimate: {self._energy_cumulative[-1]:.5f} +- {self._error[-1]:.5f}"
                outstring += f" | Block energy: {block_average_energy:.5f}"
                outstring += f" | Trial energy: {self._energy_cumulative[0]:.5f}"
                print(outstring)

            # skip equilibration blocks
            if b >= neq:
                self.update_energy_estimate(block_average_energy)
                self._reference_energy = (self._reference_energy + self._energy_cumulative[-1]) / 2


    def _run_block(self, steps_per_block, b, neq, time_step):
        block_energies = np.zeros(steps_per_block)

        for i in range(steps_per_block):

            ensemble_energies = np.zeros(len(self._walkers))

            for iwalker, walker in enumerate(self._walkers):

                self._confs.append(walker.configuration)
                local_energy = self._update_walker(walker, time_step)
                ensemble_energies[iwalker] = local_energy

                if self.force_accumulators is not None and b >= neq:
                    # accumulate samples over an ensemble
                    for fa in self.force_accumulators:
                        fa.accumulate_samples(
                            iwalker,
                            walker, 
                            self._guiding_wf, 
                            self._hamiltonian, 
                            self._reference_energy, 
                            time_step,
                            self._velocity_cutoff,
                            len(self._walkers)
                        )

            if self.force_accumulators is not None and b >= neq:
                for fa in self.force_accumulators:
                    fa.output()

            self._walkers = self._brancher.perform_branching(self._walkers)

            # perform ensemble averaging just after reconfiguration,
            # when all weights are unity
            # TODO: somehow don't recompute local energies
            ensemble_energies = [self._hamiltonian(self._guiding_wf, w.configuration)/self._guiding_wf(w.configuration) for w in self._walkers]
            ensemble_energy = np.average(ensemble_energies, weights=[w.weight for w in self._walkers])
            block_energies[i] = ensemble_energy
            self._energy_all.append(ensemble_energy)


            if b < neq:
               self._reference_energy = 0.5 * (self._reference_energy + ensemble_energy)

        return block_energies

    def _update_walker(self, walker, time_step):
        xold = walker.configuration

        # compute "old" local energy
        local_energy_old = self._hamiltonian(self._guiding_wf, xold) / walker.value

        # perform accept/reject step
        walker = self._ar.move_state(self._guiding_wf, time_step, walker)

        xnew = walker.configuration
        # compute "new" local energy
        local_energy_new = self._hamiltonian(self._guiding_wf, xnew) / walker.value

        # update walker weight and configuration
        v = walker.previous_gradient / walker.previous_value
        s = (self._reference_energy - local_energy_old) \
            * np.linalg.norm(self._velocity_cutoff(v, time_step))/np.linalg.norm(v)

        vprime = walker.gradient / walker.value
        sprime = (self._reference_energy - local_energy_new) \
             * np.linalg.norm(self._velocity_cutoff(vprime, time_step))/np.linalg.norm(vprime)

        walker.weight *= math.exp(0.5*(s + sprime)*time_step)

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