import numpy as np

from pydmc.util import flatten, velocity_cutoff


class ForcesDriftDifGfunc:

    def __init__(self, increments, num_walkers):
        self._increments = increments
        num_geos = len(increments)
        self._local_es = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._psis = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ss = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ts = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._weights = [[] for _ in range(num_walkers)]

    def accumulate_samples(self, iwalker, walker, psi, hamiltonian, eref, time_step):
        self._weights[iwalker].append(walker.weight)
        x = walker.configuration
        xprev = walker.previous_configuration

        incr = np.insert(self._increments, 0, 0)

        for geo in range(len(incr)):
            # TODO: only collect local energy on first pass to get correct S

            # setup the secondary / "deformed" wave function
            da = np.zeros(len(incr)-1)
            if geo > 0:
                da[geo-1] = incr[geo]
            psi_sec = psi.deform(da)

            psi_val = psi_sec(x)
            self._local_es[geo][iwalker].append(hamiltonian(psi_sec, x) / psi_val)
            self._psis[geo][iwalker].append(psi_val)

            local_e_prev = self._local_es[geo][iwalker][-2] if len(self._local_es[geo][iwalker]) > 1 else eref
            self._ss[geo][iwalker].append(time_step * (eref - 0.5 * (self._local_es[geo][iwalker][-1] + local_e_prev)))

            drift = velocity_cutoff(psi_sec.gradient(xprev) / psi_sec(xprev), time_step)
            self._ts[geo][iwalker].append(-np.linalg.norm(x - xprev - drift*time_step)**2 / (2*time_step))

    def compute_forces(self, steps_per_block):
        weights = flatten(self._weights)
        force = np.zeros((len(self._increments), len(weights)))
        energy = np.average(flatten(self._local_es[0]), weights=weights)

        for i, da in enumerate(self._increments):
            elocal_deriv = (np.array(flatten(self._local_es[i+1])) - np.array(flatten(self._local_es[0])))/da
            tderiv = (np.array(flatten(self._ts[i+1])) - flatten(np.array(self._ts[0])))/da
            sderiv = (np.array(flatten(self._ss[i+1])) - flatten(np.array(self._ss[0])))/da

            psi_logderiv = (np.log(np.array(flatten(self._psis[i+1]))**2) - np.log(np.array(flatten(self._psis[0]))**2))/da

            # compute partial sums of t and s derivatives
            sderiv_partial_sums = np.zeros(len(sderiv))
            tderiv_partial_sums = np.zeros(len(tderiv))

            for n in range(steps_per_block, len(sderiv)):
                sderiv_partial_sums[n-steps_per_block] = np.sum(sderiv[n-steps_per_block:n])
                tderiv_partial_sums[n-steps_per_block] = np.sum(tderiv[n-steps_per_block:n])

            # Compute Hellman-Feynman and Pulay force terms
            fhf = -elocal_deriv
            fpulay = -(np.array(flatten(self._local_es[0]) - energy) * (0*psi_logderiv + tderiv_partial_sums + sderiv_partial_sums))

            force[i, :] = fhf + fpulay

        return force, weights

