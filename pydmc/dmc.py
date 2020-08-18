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

    def run_dmc(self, time_step, num_blocks, steps_per_block, neq=1):
        # TODO: blocking, energy stabilization, variance estimation
        energies = np.zeros(num_blocks - neq)
    
        for b in range(num_blocks):

            block_energies = np.zeros(steps_per_block)

            for i in range(steps_per_block):

                ensemble_energy = 0
                total_weight = 0

                for walker in self._walkers:
                    xold = copy.deepcopy(walker.configuration)
                    # compute "old" local energy
                    local_energy_old = self._hamiltonian(self._guiding_wf, xold) / self._guiding_wf(xold)

                    # perform accept/reject step
                    acceptance_prob, xnew = self._ar.move_state(self._guiding_wf, xold, time_step)

                    # update ensemble energy and weight
                    ensemble_energy += walker.weight*local_energy_old
                    total_weight += walker.weight

                    # compute "new" local energy
                    local_energy_new = self._hamiltonian(self._guiding_wf, xnew) / self._guiding_wf(xnew)

                    # update walker weight and configuration
                    # TODO: fix the weight modification, as now the weights -> 0 as t -> inf
                    s = self._reference_energy - local_energy_old
                    sprime = self._reference_energy - local_energy_new
                    walker.weight *= math.exp((0.5*acceptance_prob*(s + sprime) + (1- acceptance_prob)*s)*time_step)
                    walker.configuration = xnew


                # update energy estimate
                ensemble_energy /= total_weight
                block_energies[i] = ensemble_energy

                self.brancher.perform_branching(self._walkers)
            
            block_average_energy = np.mean(block_energies)

            # skip equilibration blocks
            if b >= neq:
                energy_prev = self._energy_cumulative[-1]
                self._energy_cumulative.append(self._energy_cumulative[-1] + (block_average_energy - self._energy_cumulative[-1]) / (b - neq + 2))
                energy = block_average_energy
                energies[b-neq] = block_average_energy
                self._variance.append(self._variance[-1] + ((energy - energy_prev)*(energy - self._energy_cumulative[-1]))/(b - neq + 1))
                self._error.append(math.sqrt(self._variance[-1] / (b - neq + 1)))
                self._reference_energy = (self._reference_energy + block_average_energy) / 2

        return energies

    @property
    def energy_estimate(self):
        return np.array(self._energy_cumulative)

    @property
    def energy_error(self):
        return np.array(self._error)

