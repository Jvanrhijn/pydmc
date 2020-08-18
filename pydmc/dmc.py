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

    def run_dmc(self, time_step, num_blocks, steps_per_block, neq=1):
        # TODO: blocking, energy stabilization, variance estimation
        energies = np.zeros(num_blocks - neq)
    
        for b in range(num_blocks):

            block_energies = np.zeros(steps_per_block)

            for i in range(steps_per_block):

                ensemble_energy = 0
                total_weight = 0

                for walker in self._walkers:
                    xold = walker.configuration
                    # compute "old" local energy
                    local_energy_old = self._hamiltonian(self._guiding_wf, xold) / self._guiding_wf(xold)

                    # update ensemble energy and weight
                    ensemble_energy += local_energy_old
                    total_weight += walker.weight

                    # perform accept/reject step
                    xnew = self._ar.move_state(self._guiding_wf, xold)

                    # compute "new" local energy
                    local_energy_new = self._hamiltonian(self._guiding_wf, xnew) / self._guiding_wf(xnew)

                    # update walker weight and configuration
                    walker.weight *= math.exp(-time_step * ((local_energy_old + local_energy_new)/2.0 - self._reference_energy))
                    walker.configuration = xnew

                # update energy estimate
                ensemble_energy /= total_weight
                block_energies[i] = ensemble_energy

                self.brancher.perform_branching(self._walkers)
            
            block_average_energy = np.mean(block_energies)


            # skip equilibration blocks
            if b >= neq:
                energies[b-neq] = block_average_energy

            self._reference_energy = (self._reference_energy + block_average_energy) / 2

        return energies

