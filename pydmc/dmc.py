import numpy as np

from accept_reject import *
from branching import *
from walker import *
from wavefunction import *


class DMC:

    def __init__(self, walkers, brancher, ar, guiding_wf, reference_energy):
        self._walkers = walkers
        self.brancher = brancher
        self._ar = ar
        self._guiding_wf = guiding_wf
        self._reference_energy = reference_energy

    def run_dmc(self, time_step, iterations):
        # TODO: blocking, energy stabilization, variance estimation

        energies = np.zeros(iterations)
        variances = np.zeros(iterations)

        for i in range(iterations):
            ensemble_energy = 0
            total_weight = 0

            for walker in self._walkers:
                xold = walker.configuration
                # compute "old" local energy
                # TODO: allow potentials
                local_energy_old = -0.5 * self._guiding_wf.laplacian(xold) / self._guiding_wf(xold)
                # update ensemble energy and weight
                ensemble_energy += local_energy_old
                total_weight += walker.weight
                # perform accept/reject step
                xnew = self._ar.move_state(self._guiding_wf, xold)
                # compute "new" local energy
                local_energy_new = -0.5 * self._guiding_wf.laplacian(xnew) / self._guiding_wf(xnew)
                # update walker weight and configuration
                walker.weight *= math.exp(-time_step * (local_energy_old + local_energy_new)/2.0 - self._reference_energy)
                walker.configuration = xnew

            # update energy estimate
            ensemble_energy /= total_weight
            energies[i] = ensemble_energy

        return energies, variances

