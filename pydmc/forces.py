import numpy as np

from pydmc.util import flatten, velocity_cutoff
from pydmc.node_warp import *


class ForcesDriftDifGfunc:

    def __init__(self, increments, num_walkers, warp=False):
        self._increments = increments
        num_geos = len(increments)
        self._local_es = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._psis = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ss = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ts = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._weights = [[] for _ in range(num_walkers)]
        self._jacs = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._warp = warp

    def accumulate_samples(self, iwalker, walker, psi, hamiltonian, eref, time_step):
        self._weights[iwalker].append(walker.weight)
        x = walker.configuration
        xprev = walker.previous_configuration
        psival = psi(x)
        psigrad = psi.gradient(x)


        incr = np.insert(self._increments, 0, 0)

        for geo in range(len(incr)):
            # TODO: only collect local energy on first pass to get correct S

            # setup the secondary / "deformed" wave function
            da = np.zeros(len(incr)-1)
            if geo > 0:
                da[geo-1] = incr[geo]
            psi_sec = psi.deform(da)

            psisec_val = psi_sec(x)
            psisec_grad = psi_sec.gradient(x)

            if self._warp:
                xwarp, jac = node_warp(x, psival, psigrad, psisec_val, psisec_grad)
                self._jacs[geo][iwalker].append(jac)
                xprev_warp = node_warp(xprev, psival, psigrad, psisec_val, psisec_grad)[0]
            else:
                xwarp, jac = x, 1
                self._jacs[geo][iwalker].append(jac)
                xprev_warp = xprev

            self._local_es[geo][iwalker].append(hamiltonian(psi_sec, xwarp) / psi_sec(xwarp))
            self._psis[geo][iwalker].append(psisec_val)

            local_e_prev = self._local_es[geo][iwalker][-2] if len(self._local_es[geo][iwalker]) > 1 else eref
            vwarp = psi_sec.gradient(xprev_warp)/psi_sec(xprev_warp)
            vwarp_prime = psi_sec.gradient(xwarp)/psi_sec(xwarp)
            s = (eref - local_e_prev) \
                * np.linalg.norm(velocity_cutoff(vwarp, time_step))/np.linalg.norm(vwarp)
            sprime = (eref - self._local_es[geo][iwalker][-1]) \
                * np.linalg.norm(velocity_cutoff(vwarp_prime, time_step))/np.linalg.norm(vwarp_prime)

            self._ss[geo][iwalker].append(time_step * 0.5 * (s + sprime))

            drift = velocity_cutoff(psi_sec.gradient(xprev_warp) / psi_sec(xprev_warp), time_step)
            self._ts[geo][iwalker].append(-np.linalg.norm(xwarp - xprev_warp - drift*time_step)**2 / (2*time_step))

    def compute_forces(self, steps_per_block, nconf):
        #force = np.zeros(((len(self._weights), len)
        energy = np.average(flatten(self._local_es[0])[steps_per_block:], 
                            weights=flatten(self._weights)[steps_per_block:])

        force = np.zeros((len(self._increments), len(self._weights), len(self._weights[0])))
        elocal = np.array(self._local_es)
        ts = np.array(self._ts)
        ss = np.array(self._ss)
        jacs = np.array(self._jacs)

        for i, da in enumerate(self._increments):
            elocal_deriv = (elocal[i+1] - elocal[0])/da
            tderiv = (ts[i+1] - ts[0])/da
            sderiv = (ss[i+1] - ss[0])/da

            jderiv = (np.log(np.abs(jacs[i+1])) - np.log(np.abs(jacs[0])))/da
        
            # compute partial sums of t and s derivatives
            sderiv_partial_sums = np.zeros(sderiv.shape)
            tderiv_partial_sums = np.zeros(sderiv.shape)

            for n in range(steps_per_block, len(sderiv)):
                sderiv_partial_sums[:, n] = np.sum(sderiv[:, n-steps_per_block:n], axis=1)
                tderiv_partial_sums[:, n] = np.sum(tderiv[:, n-steps_per_block:n], axis=1)


            # Compute Hellman-Feynman and Pulay force terms
            # TODO: align properly
            fhf = -elocal_deriv 
            fpulay = -(elocal[0] - energy) \
                * (tderiv_partial_sums + sderiv_partial_sums + jderiv) # + psi_logderiv[:-steps_per_block]))

            force[i, :, :] = fhf + fpulay

        w = np.array(self._weights)[:, steps_per_block:nconf]
        flout = np.zeros((len(self._increments), w.shape[1]))
        fave = np.zeros(len(self._increments))

        for i in range(len(self._increments)):
            flout[i] = np.average(force[i, :, steps_per_block:nconf], weights=w, axis=0)
            fave[i] = np.average(force[i, :, steps_per_block:nconf], weights=w)

        return flout, fave


class ForcesVD:

    def __init__(self, increments, num_walkers, warp=False):
        self._increments = increments
        num_geos = len(increments)
        self._local_es = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._psis = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ss = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._weights = [[] for _ in range(num_walkers)]
        self._jacs = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._warp = warp

    def accumulate_samples(self, iwalker, walker, psi, hamiltonian, eref, time_step):
        self._weights[iwalker].append(walker.weight)
        x = walker.configuration
        xprev = walker.previous_configuration

        psival = psi(x)
        psigrad = psi.gradient(x)

        incr = np.insert(self._increments, 0, 0)

        for geo in range(len(incr)):
            # TODO: only collect local energy on first pass to get correct S

            # setup the secondary / "deformed" wave function
            da = np.zeros(len(incr)-1)
            if geo > 0:
                da[geo-1] = incr[geo]
            psi_sec = psi.deform(da)

            psisec_val = psi_sec(x)
            psisec_grad = psi_sec.gradient(x)

            if self._warp:
                xwarp, jac = node_warp(x, psival, psigrad, psisec_val, psisec_grad)
                self._jacs[geo][iwalker].append(jac)
                xprev_warp = node_warp(xprev, psival, psigrad, psisec_val, psisec_grad)[0]
            else:
                xwarp, jac = x, 1
                self._jacs[geo][iwalker].append(jac)
                xprev_warp = xprev

            self._local_es[geo][iwalker].append(hamiltonian(psi_sec, xwarp) / psi_sec(xwarp))
            local_e_prev = self._local_es[geo][iwalker][-2] if len(self._local_es[geo][iwalker]) > 1 else eref

            vwarp = psi_sec.gradient(xprev_warp)/psi_sec(xprev_warp)
            vwarp_prime = psi_sec.gradient(xwarp)/psi_sec(xwarp)
            s = (eref - local_e_prev) \
                * np.linalg.norm(velocity_cutoff(vwarp, time_step))/np.linalg.norm(vwarp)
            sprime = (eref - self._local_es[geo][iwalker][-1]) \
                * np.linalg.norm(velocity_cutoff(vwarp_prime, time_step))/np.linalg.norm(vwarp_prime)

            self._ss[geo][iwalker].append(time_step * 0.5 * (s + sprime))

            self._psis[geo][iwalker].append(psisec_val)

            #self._ss[geo][iwalker].append(time_step * (eref - 0.5 * (self._local_es[geo][iwalker][-1] + local_e_prev)))

#    def compute_forces(self, steps_per_block):
#        weights = flatten(self._weights)
#        force = np.zeros((len(self._increments), len(weights)))
#        energy = np.average(flatten(self._local_es[0]), weights=weights)
#
#        for i, da in enumerate(self._increments):
#            elocal_deriv = (np.array(flatten(self._local_es[i+1])) - np.array(flatten(self._local_es[0])))/da
#            sderiv = (np.array(flatten(self._ss[i+1])) - flatten(np.array(self._ss[0])))/da
#
#            psi_logderiv = (np.log(np.abs(np.array(flatten(self._psis[i+1])))) - np.log(np.abs(np.array(flatten(self._psis[0])))))/da
#
#            jprimary = np.array(flatten(self._jacs[0]))
#            jderiv = (np.log(np.abs(np.array(flatten(self._jacs[i+1])))) - np.log(np.abs(jprimary)))/da
#
#            # compute partial sums of t and s derivatives
#            sderiv_partial_sums = np.zeros(len(sderiv))
#
#            for n in range(steps_per_block, len(sderiv)):
#                sderiv_partial_sums[n-steps_per_block] = np.sum(sderiv[n-steps_per_block:n])
#
#            # Compute Hellman-Feynman and Pulay force terms
#            fhf = -elocal_deriv
#            fpulay = -(np.array(flatten(self._local_es[0]) - energy) * (2*psi_logderiv + sderiv_partial_sums + jderiv))
#
#            force[i, :] = fhf + fpulay
#
#        return force, weights

    #def compute_forces(self, steps_per_block):
    #    #force = np.zeros(((len(self._weights), len)
    #    energy = np.average(flatten(self._local_es[0])[steps_per_block:], 
    #                        weights=flatten(self._weights)[steps_per_block:])

    #    force = np.zeros((len(self._weights), len(self._increments), len(self._weights[0])-steps_per_block))
    #    # take derivatives per walker, average at the end
    #    weights_tot = 0
    #    for iw in range(len(self._weights)):
    #        for i, da in enumerate(self._increments):
    #            elocal_deriv = (np.array(self._local_es[i+1][iw]) - np.array(self._local_es[0][iw]))/da
    #            sderiv = (np.array(self._ss[i+1][iw]) - np.array(self._ss[0][iw]))/da

    #            jprimary = np.array(self._jacs[0][iw])
    #            jderiv = (np.log(np.abs(np.array(self._jacs[i+1][iw]))) - np.log(np.abs(jprimary)))/da
    #        
    #            psis_prim = np.array(self._psis[0][iw])
    #            psis_sec = np.array(self._psis[i+1][iw])
    #            psi_logderiv = (np.log(np.abs(psis_sec)) - np.log(np.abs(psis_prim)))/da

    #            # compute partial sums of t and s derivatives
    #            sderiv_partial_sums = np.zeros(len(sderiv)-steps_per_block)

    #            for n in range(steps_per_block, len(sderiv)):
    #                sderiv_partial_sums[n-steps_per_block] = np.sum(sderiv[n-steps_per_block:n])

    #            # Compute Hellman-Feynman and Pulay force terms
    #            fhf = -elocal_deriv[steps_per_block:]
    #            fpulay = -(np.array(self._local_es[0][iw][steps_per_block:] - energy) \
    #                * (2*psi_logderiv[steps_per_block:] + sderiv_partial_sums + jderiv[steps_per_block:]))

    #            force[iw, i, :] = fhf + fpulay
    #        weights_tot += np.array(self._weights[iw])[steps_per_block:]

    #    flout = np.zeros(force.shape[1:])
    #    fave = np.zeros(len(self._increments))
    #    w = np.array(self._weights)

    #    for i in range(len(self._increments)):
    #        flout[i] = np.average(force[:, i, :], weights=w[:, steps_per_block:], axis=0)
    #        fave[i] = np.average(force[:, i, :], weights=w[:, steps_per_block:])

    #    return flout, fave

    def compute_forces(self, steps_per_block, nconf):
        #force = np.zeros(((len(self._weights), len)
        energy = np.average(flatten(self._local_es[0])[steps_per_block:], 
                            weights=flatten(self._weights)[steps_per_block:])

        force = np.zeros((len(self._increments), len(self._weights), len(self._weights[0])))
        elocal = np.array(self._local_es)
        ss = np.array(self._ss)
        psis = np.array(self._psis)
        jacs = np.array(self._jacs)

        for i, da in enumerate(self._increments):
            elocal_deriv = (elocal[i+1] - elocal[0])/da
            sderiv = (ss[i+1] - ss[0])/da
            psideriv = (np.log(np.abs(psis[i+1])) - np.log(np.abs(psis[0])))/da
            jderiv = (np.log(np.abs(jacs[i+1])) - np.log(np.abs(jacs[0])))/da
        
            # compute partial sums of t and s derivatives
            sderiv_partial_sums = np.zeros(sderiv.shape)

            for n in range(steps_per_block, len(sderiv)):
                sderiv_partial_sums[:, n] = np.sum(sderiv[:, n-steps_per_block:n], axis=1)

            # Compute Hellman-Feynman and Pulay force terms
            # TODO: align properly
            fhf = -elocal_deriv 
            fpulay = -(elocal[0] - energy) \
                * (2*psideriv + sderiv_partial_sums + jderiv) # + psi_logderiv[:-steps_per_block]))

            force[i, :, :] = fhf + fpulay

        w = np.array(self._weights)[:, steps_per_block:nconf]
        flout = np.zeros((len(self._increments), w.shape[1]))
        fave = np.zeros(len(self._increments))

        for i in range(len(self._increments)):
            flout[i] = np.average(force[i, :, steps_per_block:nconf], weights=w, axis=0)
            fave[i] = np.average(force[i, :, steps_per_block:nconf], weights=w)

        return flout, fave
