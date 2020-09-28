import numpy as np
import pprint as pp
import tqdm
import h5py

from pydmc.util import flatten, munu, velocity_cutoff_umrigar
from pydmc.node_warp import *


class ForcesVMCSorella:

    def __init__(self, increments, epsilon=1e-2):
        self._increments = increments
        num_geos = len(increments)
        self._local_es = [[] for _ in range(num_geos+1)]
        self._psis = [[] for _ in range(num_geos+1)]
        self._distrs = [[] for _ in range(num_geos+1)]

        self._confs = []

        self._epsilon = epsilon

    def accumulate_samples(self, conf, psi, hamiltonian):

        self._confs.append(conf)
        x = conf
        incr = np.insert(self._increments, 0, 0)

        for geo in range(len(incr)):
            # setup the secondary / "deformed" wave function
            da = np.zeros(len(incr)-1)
            if geo > 0:
                da[geo-1] = incr[geo]
            psi_sec = psi.deform(da)

            psisec_val = psi_sec(x)
            psisec_grad = psi_sec.gradient(x)

            self._local_es[geo].append(hamiltonian(psi_sec, x) / psisec_val)
            self._psis[geo].append(psisec_val)
            self._distrs[geo].append(self.distribution(psisec_val, psisec_grad))

    def compute_forces(self, steps_per_block, nconf):
        forcel_hf = np.zeros((len(self._increments), len(self._local_es[0])))
        forcel_pulay = np.zeros((len(self._increments), len(self._local_es[0])))

        elocal = np.array(self._local_es)
        psis = np.array(self._psis)
        energy = np.mean(elocal[0])
        distrs = np.array(self._distrs)

        weights = psis[0]**2 / distrs[0]

        for i, da in enumerate(self._increments):
            elocal_deriv = (elocal[i+1] - elocal[0])/da
            psideriv = (np.log(np.abs(psis[i+1]**2)) - np.log(np.abs(psis[0]**2)))/da

            #Compute Hellman-Feynman and Pulay force terms
            forcel_hf[i, :] = -elocal_deriv
            forcel_pulay[i, :] = -(elocal[0, :] - energy) * psideriv

        fhf = np.average(forcel_hf[0], weights=weights)

        fpulay = np.average(forcel_pulay[0], weights=weights)

        return forcel_hf*weights, forcel_pulay*weights, \
            fhf, fpulay

    def distribution(self, psi, psigrad):
        # compute new distribution according to Sorella
        d = node_distance(psigrad, psi)
        deps = d if d > self._epsilon \
            else self._epsilon*(d/self._epsilon)**(d/self._epsilon)
        return psi**2 * (deps/d)**2


class VMCForcesInput:

    def compute_forces(self, fpath):
        #data = self._retrieve_data(fpath)
        data = h5py.File(fpath, "r")

        local_energy = data["Local energy"].value[1:]
        energy = np.mean(local_energy)

        el_grad = data["grad_a E_L"].value[1:]
        el_grad_warp = data["grad_a E_L (warp)"].value[1:]

        logpsi_grad = data["grad_a log Psi"].value[1:]
        logpsi_grad_warp = data["grad_a log Psi (warp)"].value[1:]

        el_times_logpsi_grad = data["E_L * grad_a log Psi"].value[1:]
        el_times_logpsi_grad_warp = data["E_L * grad_a log Psi (warp)"].value[1:]
        
        logjac_grad = data["grad_a log Jacobian"].value[1:]
        el_times_logjac_grad = data["E_L * grad_a log Jacobian"].value[1:]

        # TODO: need to do this for sequence of epsilon
        #pathak = data["Pathak regularizer"][:, 0]

        force_hf = -el_grad
        force_hf_warp = -el_grad_warp

        # pulay force
        force_pulay = -(2 * (el_times_logpsi_grad - energy * logpsi_grad))
        force_pulay_warp = -(2 * (el_times_logpsi_grad_warp - energy * logpsi_grad_warp) \
                                + (el_times_logjac_grad - energy * logjac_grad))
        #force_pulay_warp = -(local_energy - energy) * (2*psilogderiv_warp + jac_deriv)

        # TODO:
        force_pulay_pathak = -(local_energy - energy) * 0

        data.close()

        return force_hf.flatten(), force_pulay.flatten(), force_hf_warp.flatten(), force_pulay_warp.flatten(), force_pulay_pathak.flatten()


class DMCForcesInput:

    def _retrieve_data(self, fpath):
        from numpy import array
        num_lines = sum(1 for line in open(fpath))
        data = {}
        with open(fpath, "r") as file:
            for i, line in tqdm.tqdm(enumerate(file), total=num_lines):
                # this makes me want to cry
                if i == 0:
                    self._num_walkers = int(line.split()[3])
                    self._steps_per_block = int(line.split()[-1])
                    continue
                ldata = eval(line)
                if not data:
                    for key, value in ldata.items():
                        data[key] = [value]
                else:
                    for key, value in ldata.items():
                        data[key].append(value)
        for key, value in data.items():
            data[key] = np.array(value)
        return data

    def compute_forces(self, fpath, wagner=False):
        data = self._retrieve_data(fpath)
        energy = np.mean(data["Local energy"])
        weights = data["Weight"]

        # Get Green's function derivatives
        sderiv_sum = data["grad_a S"]        
        sderiv_sum_warp = data["grad_a S (warp)"]        

        tderiv_sum_nocutoff = data["grad_a T (no cutoff)"]        
        tderiv_sum_warp_nocutoff = data["grad_a T (warp, no cutoff)"]        

        tderiv_sum = data["grad_a T"]        
        tderiv_sum_warp = data["grad_a T (warp)"]        

        # Get Jacobian (sum) derivative
        jderiv_sum = data["grad_a sum Log Jacobian"]
        jac_logderiv = data["grad_a Log Jacobian"]

        # Get products of E_L and Green's function derivatives
        el_times_sderiv_sum = data["E_L * grad_a S"]        
        el_times_sderiv_sum_warp = data["E_L * grad_a S (warp)"]        

        el_times_tderiv_sum_nocutoff = data["E_L * grad_a T (no cutoff)"]        
        el_times_tderiv_sum_warp_nocutoff = data["E_L * grad_a T (warp, no cutoff)"]        

        el_times_tderiv_sum = data["E_L * grad_a T"]        
        el_times_tderiv_sum_warp = data["E_L * grad_a T (warp)"]        

        # Get product of E_L and Jacobian (sum) derivative
        el_times_jderiv_sum = data["E_L * grad_a sum Log Jacobian"]
        el_times_jac_logderiv = data["E_L * grad_a Log Jacobian"]

        # Get local e derivative
        local_e_deriv = data["grad_a E_L"]
        local_e_deriv_warp = data["grad_a E_L (warp)"]

        # Get psi derivative
        psilogderiv = data["grad_a Log Psi"]
        psilogderiv_warp = data["grad_a Log Psi (warp)"]

        el_times_psilogderiv = data["E_L * grad_a Log Psi"]
        el_times_psilogderiv_warp = data["E_L * grad_a Log Psi (warp)"]

        # Hellmann-Feynman force
        force_hf = -local_e_deriv
        force_hf_warp = -local_e_deriv_warp

        # Pulay force
        force_pulay_exact = -(
                    el_times_tderiv_sum - energy*tderiv_sum \
                +   el_times_sderiv_sum - energy*sderiv_sum \
                )

        force_pulay_exact_warp = -(
                    el_times_tderiv_sum_warp - energy*tderiv_sum_warp \
                +   el_times_sderiv_sum_warp - energy*sderiv_sum_warp \
                +   el_times_jderiv_sum - energy*jderiv_sum \
                )

        force_pulay_exact_nocutoff = -(
                    el_times_tderiv_sum_nocutoff - energy*tderiv_sum_nocutoff \
                +   el_times_sderiv_sum - energy*sderiv_sum \
                )

        force_pulay_exact_warp_nocutoff = -(
                    el_times_tderiv_sum_warp_nocutoff - energy*tderiv_sum_warp_nocutoff \
                +   el_times_sderiv_sum_warp - energy*sderiv_sum_warp \
                +   el_times_jderiv_sum - energy*jderiv_sum \
                )

        force_pulay_vd = -(
                2 * (el_times_psilogderiv - energy*psilogderiv) \
                +    el_times_sderiv_sum - energy*sderiv_sum
                )

        force_pulay_vd_warp = -(
                2 * (el_times_psilogderiv_warp - energy*psilogderiv_warp) \
                +    el_times_sderiv_sum_warp - energy*sderiv_sum_warp \
                +    el_times_jac_logderiv - energy * jac_logderiv \
                )

        return force_hf, \
               force_hf_warp, \
               force_pulay_exact, \
               force_pulay_exact_warp, \
               force_pulay_vd, \
               force_pulay_vd_warp, \
               force_pulay_exact_nocutoff, \
               force_pulay_exact_warp_nocutoff, \
               self._steps_per_block, \
               weights

