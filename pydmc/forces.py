import numpy as np

from pydmc.util import flatten, munu
from pydmc.node_warp import *


class ForcesExactAndVD:

    def __init__(self, increments, num_walkers, cutoff=lambda d: (0, 0, 0)):
        self._increments = increments
        num_geos = len(increments)
        self._local_es = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._psis = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ss = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ts = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._weights = [[] for _ in range(num_walkers)]
        self._confs = [[] for _ in range(num_walkers)]
        self._jacs = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]

        self._local_es_warp = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._psis_warp = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ss_warp = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]
        self._ts_warp = [[[] for _ in range(num_walkers)] for _ in range(num_geos+1)]

        self._cutoff = cutoff

    def accumulate_samples(self, iwalker, walker, psi, hamiltonian, eref, time_step, velocity_cutoff):

        self._weights[iwalker].append(walker.weight)
        self._confs[iwalker].append(walker.configuration)
        x = walker.configuration
        xprev = walker.previous_configuration

        psival = psi(x)
        psigrad = psi.gradient(x)

        psival_prev = psi(xprev)
        psigrad_prev = psi.gradient(xprev)

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
            psisec_val_prev = psi_sec(xprev)
            psisec_grad_prev = psi_sec.gradient(xprev)

            xwarp, jac, j \
                = node_warp(x, psival, psigrad, psisec_val, psisec_grad, cutoff=self._cutoff)
            xprev_warp, jac_prev, jprev \
                 = node_warp(xprev, psival_prev, psigrad_prev, psisec_val_prev, psisec_grad_prev, cutoff=self._cutoff)

            psisec_val_warp = psi_sec(xwarp)
            psisec_grad_warp = psi_sec.gradient(xwarp)
            psisec_val_prev_warp = psi_sec(xprev_warp)
            psisec_grad_prev_warp = psi_sec.gradient(xprev_warp)

            self._jacs[geo][iwalker].append(jac_prev)

            self._local_es[geo][iwalker].append(hamiltonian(psi_sec, x) / psisec_val)
            self._local_es_warp[geo][iwalker].append(hamiltonian(psi_sec, xwarp) / psisec_val_warp)
            
            self._psis[geo][iwalker].append(psisec_val)
            self._psis_warp[geo][iwalker].append(psisec_val_warp)

            local_e_prev = self._local_es[geo][iwalker][-2] if len(self._local_es[geo][iwalker]) > 1 else eref
            local_e_prev_warp = self._local_es_warp[geo][iwalker][-2] if len(self._local_es_warp[geo][iwalker]) > 1 else eref

            v = psisec_grad_prev/psisec_val_prev
            vprime = psisec_grad/psisec_val
            s = (eref - local_e_prev) \
                * np.linalg.norm(velocity_cutoff(v, time_step))/np.linalg.norm(v)
            sprime = (eref - self._local_es[geo][iwalker][-1]) \
                * np.linalg.norm(velocity_cutoff(vprime, time_step))/np.linalg.norm(vprime)

            vwarp = psisec_grad_prev_warp/psisec_val_prev_warp
            vprime_warp = psisec_grad_warp/psisec_val_warp
            swarp = (eref - local_e_prev_warp) \
                * np.linalg.norm(velocity_cutoff(vwarp, time_step))/np.linalg.norm(vwarp)
            sprime_warp = (eref - self._local_es_warp[geo][iwalker][-1]) \
                * np.linalg.norm(velocity_cutoff(vprime_warp, time_step))/np.linalg.norm(vprime_warp)

            self._ss[geo][iwalker].append(time_step * 0.5 * (s + sprime))
            self._ss_warp[geo][iwalker].append(time_step * 0.5 * (swarp + sprime_warp))

            drift = velocity_cutoff(psi_sec.gradient(xprev) / psi_sec(xprev), time_step)
            drift_warp = velocity_cutoff(psisec_grad_prev_warp / psisec_val_prev_warp, time_step)
            # if last move was rejected, set T = 0  rather than -(Vt)^2/2t.
            # TODO: think about how T should look in transformed space
            if np.all(x == xprev):
                self._ts[geo][iwalker].append(0)
                self._ts_warp[geo][iwalker].append(0)
            else:
                u = x - xprev - drift*time_step
                uwarp = xwarp - xprev_warp - drift_warp*time_step
                self._ts[geo][iwalker].append(-np.linalg.norm(u)**2 / (2*time_step))
                self._ts_warp[geo][iwalker].append(-np.linalg.norm(uwarp)**2 / (2*time_step))

    def compute_forces(self, steps_per_block, nconf):
        forcel_hf = np.zeros((len(self._increments), len(self._weights), len(self._weights[0])))
        forcel_hf_warp = np.zeros((len(self._increments), len(self._weights), len(self._weights[0])))

        forcel_pulay_exact = np.zeros((len(self._increments), len(self._weights), len(self._weights[0])))
        forcel_pulay_exact_warp = np.zeros((len(self._increments), len(self._weights), len(self._weights[0])))
        forcel_pulay_vd = np.zeros((len(self._increments), len(self._weights), len(self._weights[0])))
        forcel_pulay_vd_warp = np.zeros((len(self._increments), len(self._weights), len(self._weights[0])))

        elocal = np.array(self._local_es)
        ss = np.array(self._ss)
        ts = np.array(self._ts)
        psis = np.array(self._psis)
        w = np.array(self._weights)[:, steps_per_block:nconf]

        elocal_warp = np.array(self._local_es_warp)
        ss_warp = np.array(self._ss_warp)
        ts_warp = np.array(self._ts_warp)
        psis_warp = np.array(self._psis_warp)
        jacs = np.array(self._jacs)

        energy = np.average(elocal[0, :, steps_per_block:nconf], weights=w)

        for i, da in enumerate(self._increments):
            elocal_deriv = (elocal[i+1] - elocal[0])/da
            elocal_deriv_warp = (elocal_warp[i+1] - elocal_warp[0])/da

            sderiv = (ss[i+1] - ss[0])/da
            sderiv_warp = (ss_warp[i+1] - ss_warp[0])/da

            tderiv = (ts[i+1] - ts[0])/da
            tderiv_warp = (ts_warp[i+1] - ts_warp[0])/da

            psideriv = (np.log(np.abs(psis[i+1]**2)) - np.log(np.abs(psis[0]**2)))/da
            psideriv_warp = (np.log(np.abs(psis_warp[i+1]**2)) - np.log(np.abs(psis_warp[0]**2)))/da

            jderiv = (np.log(np.abs(jacs[i+1])) - np.log(np.abs(jacs[0])))/da

            # compute partial sums of t and s derivatives
            sderiv_partial_sums = np.zeros(sderiv.shape)
            tderiv_partial_sums = np.zeros(sderiv.shape)
            jderiv_partial_sums = np.zeros(jderiv.shape)

            sderiv_partial_sums_warp = np.zeros(sderiv.shape)
            tderiv_partial_sums_warp = np.zeros(sderiv.shape)

            for n in range(steps_per_block, sderiv.shape[1]):
              sderiv_partial_sums[:, n] = np.sum(sderiv[:, n-steps_per_block:n], axis=1)
              tderiv_partial_sums[:, n] = np.sum(tderiv[:, n-steps_per_block:n], axis=1)

              sderiv_partial_sums_warp[:, n] = np.sum(sderiv_warp[:, n-steps_per_block:n], axis=1)
              tderiv_partial_sums_warp[:, n] = np.sum(tderiv_warp[:, n-steps_per_block:n], axis=1)

              jderiv_partial_sums[:, n] = np.sum(jderiv[:, n-steps_per_block:n], axis=1)

            #Compute Hellman-Feynman and Pulay force terms
            forcel_hf[i, :, :] = -elocal_deriv 
            forcel_hf_warp[i, :, :] = -elocal_deriv_warp

            forcel_pulay_exact[i, :, :] = -(elocal[0] - energy) \
                * (tderiv_partial_sums + sderiv_partial_sums)

            forcel_pulay_exact_warp[i, :, :] = -(elocal[0] - energy) \
                * (tderiv_partial_sums_warp + sderiv_partial_sums_warp + jderiv_partial_sums)

            forcel_pulay_vd[i, :, :] = -(elocal[0, :, :] - energy) \
                * (psideriv + sderiv_partial_sums)

            forcel_pulay_vd_warp[i, :, :] = -(elocal[0, :, :] - energy) \
                * (psideriv_warp + sderiv_partial_sums_warp + jderiv)


        flhf_out = np.zeros((len(self._increments), w.shape[1]))
        flhf_warp_out = np.zeros((len(self._increments), w.shape[1]))

        flpulay_exact_out = np.zeros((len(self._increments), w.shape[1]))
        flpulay_exact_warp_out = np.zeros((len(self._increments), w.shape[1]))
        flpulay_vd_out = np.zeros((len(self._increments), w.shape[1]))
        flpulay_vd_warp_out = np.zeros((len(self._increments), w.shape[1]))

        fhf = np.zeros(len(self._increments))
        fhf_warp = np.zeros(len(self._increments))

        fpulay_exact = np.zeros(len(self._increments))
        fpulay_exact_warp = np.zeros(len(self._increments))
        fpulay_vd = np.zeros(len(self._increments))
        fpulay_vd_warp = np.zeros(len(self._increments))

        for i in range(len(self._increments)):
            flhf_out[i] = np.average(forcel_hf[i, :, steps_per_block:nconf], weights=w, axis=0)
            flhf_warp_out[i] = np.average(forcel_hf_warp[i, :, steps_per_block:nconf], weights=w, axis=0)

            flpulay_exact_out[i] = np.average(forcel_pulay_exact[i, :, steps_per_block:nconf], weights=w, axis=0)
            flpulay_exact_warp_out[i] = np.average(forcel_pulay_exact_warp[i, :, steps_per_block:nconf], weights=w, axis=0)
            flpulay_vd_out[i] = np.average(forcel_pulay_vd[i, :, steps_per_block:nconf], weights=w, axis=0)
            flpulay_vd_warp_out[i] = np.average(forcel_pulay_vd_warp[i, :, steps_per_block:nconf], weights=w, axis=0)

            fhf[i] = np.average(forcel_hf[i, :, steps_per_block:nconf], weights=w)
            fhf_warp[i] = np.average(forcel_hf_warp[i, :, steps_per_block:nconf], weights=w)

            fpulay_exact[i] = np.average(forcel_pulay_exact[i, :, steps_per_block:nconf], weights=w)
            fpulay_exact_warp[i] = np.average(forcel_pulay_exact_warp[i, :, steps_per_block:nconf], weights=w)
            fpulay_vd[i] = np.average(forcel_pulay_vd[i, :, steps_per_block:nconf], weights=w)
            fpulay_vd_warp[i] = np.average(forcel_pulay_vd_warp[i, :, steps_per_block:nconf], weights=w)

        return flhf_out, flhf_warp_out, flpulay_exact_out, flpulay_exact_warp_out, \
            flpulay_vd_out, flpulay_vd_warp_out, fhf, fhf_warp, \
            fpulay_exact, fpulay_exact_warp, fpulay_vd, fpulay_vd_warp, w


class ForcesVMC:

    def __init__(self, increments, cutoff=lambda d: (0, 0, 0)):
        self._increments = increments
        num_geos = len(increments)
        self._local_es = [[] for _ in range(num_geos+1)]
        self._psis = [[] for _ in range(num_geos+1)]

        self._local_es_warp = [[] for _ in range(num_geos+1)]
        self._psis_warp = [[] for _ in range(num_geos+1)]

        self._confs = []
        self._jacs = [[] for _ in range(num_geos+1)]

        self._cutoff = cutoff

    def accumulate_samples(self, conf, psi, hamiltonian):

        self._confs.append(conf)
        x = conf
        psival = psi(x)
        psigrad = psi.gradient(x)

        incr = np.insert(self._increments, 0, 0)

        for geo in range(len(incr)):
            # setup the secondary / "deformed" wave function
            da = np.zeros(len(incr)-1)
            if geo > 0:
                da[geo-1] = incr[geo]
            psi_sec = psi.deform(da)

            psisec_val = psi_sec(x)
            psisec_grad = psi_sec.gradient(x)

            xwarp, jac, j \
                = node_warp(x, psival, psigrad, psisec_val, psisec_grad, cutoff=self._cutoff)

            psisec_val_warp = psi_sec(xwarp)

            self._jacs[geo].append(jac)

            self._local_es[geo].append(hamiltonian(psi_sec, x) / psisec_val)
            self._local_es_warp[geo].append(hamiltonian(psi_sec, xwarp) / psisec_val_warp)
            
            self._psis[geo].append(psisec_val)
            self._psis_warp[geo].append(psisec_val_warp)

    def compute_forces(self, steps_per_block, nconf):
        forcel_hf = np.zeros((len(self._increments), len(self._local_es[0])))
        forcel_hf_warp = np.zeros((len(self._increments), len(self._local_es[0])))

        forcel_pulay = np.zeros((len(self._increments), len(self._local_es[0])))
        forcel_pulay_warp = np.zeros((len(self._increments), len(self._local_es[0])))

        elocal = np.array(self._local_es)
        psis = np.array(self._psis)

        elocal_warp = np.array(self._local_es_warp)
        psis_warp = np.array(self._psis_warp)

        jacs = np.array(self._jacs)

        energy = np.mean(elocal[0])

        for i, da in enumerate(self._increments):
            elocal_deriv = (elocal[i+1] - elocal[0])/da
            elocal_deriv_warp = (elocal_warp[i+1] - elocal_warp[0])/da

            psideriv = (np.log(np.abs(psis[i+1]**2)) - np.log(np.abs(psis[0]**2)))/da
            psideriv_warp = (np.log(np.abs(psis_warp[i+1]**2)) - np.log(np.abs(psis_warp[0]**2)))/da

            jderiv = (np.log(np.abs(jacs[i+1])) - np.log(np.abs(jacs[0])))/da

            #Compute Hellman-Feynman and Pulay force terms
            forcel_hf[i, :] = -elocal_deriv 
            forcel_hf_warp[i, :] = -elocal_deriv_warp

            forcel_pulay[i, :] = -(elocal[0, :] - energy) * psideriv

            forcel_pulay_warp[i, :] = -(elocal[i, :] - energy) \
                * (psideriv_warp + jderiv)

        fhf = np.mean(forcel_hf, axis=1)
        fhf_warp = np.mean(forcel_hf_warp, axis=1)

        fpulay = np.mean(forcel_pulay, axis=1)
        fpulay_warp = np.mean(forcel_pulay_warp, axis=1)

        return forcel_hf, forcel_hf_warp, forcel_pulay, forcel_pulay_warp, \
            fhf, fhf_warp, fpulay, fpulay_warp