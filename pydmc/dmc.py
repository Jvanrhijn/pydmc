import copy
import math
import numpy as np

from pydmc.accept_reject import *
from pydmc.branching import *
from pydmc.walker import *
from pydmc.wavefunction import *


class DMC:

    def __init__(self, hamiltonian, walkers, brancher, ar, guiding_wf, reference_energy):
        self._hamiltonian = hamiltonian
        self._walkers = walkers
        self.brancher = brancher
        self._ar = ar
        self._guiding_wf = guiding_wf
        self._reference_energy = reference_energy
        self._energy_cumulative = [reference_energy]
        self._variance = [0.0]
        self._error = [0.0]
        self._observables = {}
        self._observables_cumulative = {}
        self._observables_variance = {}
        self._observables_error = {}

    def add_observable(self, name, observable):
        obs_cum = 0
        for walker in self._walkers:
            obs_cum += observable(self._guiding_wf, walker.configuration)/len(self._walkers)
        self._observables[name] = observable
        self._observables_cumulative[name] = [obs_cum]
        self._observables_variance[name] = [0.0]
        self._observables_error[name] = [0.0]

    def run_dmc(self, time_step, num_blocks, steps_per_block, neq=1):
        # TODO: blocking, energy stabilization, variance estimation
        energies = np.zeros(num_blocks - neq)
    
        for b in range(num_blocks):

            block_energies = np.zeros(steps_per_block)
            observable_blocks = {name: np.zeros(steps_per_block) for name in self._observables}

            for i in range(steps_per_block):

                ensemble_energy = 0
                total_weight = 0

                # set up ensemble average for each observable
                observable_ensembles = {name: 0 for name in self._observables}

                for walker in self._walkers:
                    xold = copy.deepcopy(walker.configuration)
                    wf_value_old = self._guiding_wf(xold)

                    # compute "old" local energy
                    local_energy_old = self._hamiltonian(self._guiding_wf, xold) / wf_value_old

                    # update ensembles and weights
                    for name in observable_ensembles:
                        observable_ensembles[name] += self._observables[name](self._guiding_wf, xold) / wf_value_old

                    ensemble_energy += walker.weight*local_energy_old
                    total_weight += walker.weight

                    # perform accept/reject step
                    acceptance_prob, xnew = self._ar.move_state(self._guiding_wf, xold, time_step)

                    # compute "new" local energy
                    local_energy_new = self._hamiltonian(self._guiding_wf, xnew) / self._guiding_wf(xnew)

                    # update walker weight and configuration
                    s = self._reference_energy - local_energy_old
                    sprime = self._reference_energy - local_energy_new
                    walker.weight *= math.exp((0.5*acceptance_prob*(s + sprime) + (1- acceptance_prob)*s)*time_step)
                    walker.configuration = xnew

                # update observable estimates
                for name in observable_ensembles:
                    observable_ensembles[name] /= total_weight
                    observable_blocks[name][i] = observable_ensembles[name]

                ensemble_energy /= total_weight
                block_energies[i] = ensemble_energy

                self.brancher.perform_branching(self._walkers)
            
            # skip equilibration blocks
            if b >= neq:
                # compute block averages of observables after equilibration
                block_average_energy = np.mean(block_energies)
                self.update_energy_estimate(block_average_energy)
                self._reference_energy = (self._reference_energy + self._energy_cumulative[-1]) / 2
                block_average_obs = {name: np.mean(value) for name, value in observable_blocks.items()}
                self.update_observable_estimate(block_average_obs)

        return energies

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

    def update_observable_estimate(self, obs_new):
        for name, value in obs_new.items():
            idx = len(self._observables_cumulative[name]) - 1
            obs_prev = self._observables_cumulative[name][-1]
            self._observables_cumulative[name].append(obs_prev + (value - obs_prev) / (idx + 1))
            self._observables_variance[name].append(
                self._observables_variance[name][-1] \
                       + ((value - obs_prev)*(value - self._observables_cumulative[name][-1]))/(idx + 1)
            )
            self._observables_error[name].append(math.sqrt(self._observables_variance[name][-1] / (idx + 1)))