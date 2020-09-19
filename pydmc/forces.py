import numpy as np
import pprint as pp
import tqdm

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

    def _retrieve_data(self, fpath):
        from numpy import array
        num_lines = sum(1 for line in open(fpath))
        data = {}
        with open(fpath, "r") as file:
            for line in tqdm.tqdm(file, total=num_lines):
                # this makes me want to cry
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

    def compute_forces(self, fpath):
        data = self._retrieve_data(fpath)

        da = data["da"].flatten()
        local_energy = data["Local energy"]
        local_energy_sec = data["Local energy (secondary)"]
        local_energy_sec_warp = data["Local energy (secondary, warp)"]

        psi = data["Psi"]
        psi_sec = data["Psi (secondary)"]
        psi_sec_warp = data["Psi (secondary, warp)"]
        
        jac = data["Jacobian"]

        # compute local e derivative
        local_e_deriv = (local_energy_sec - local_energy) / da
        local_e_deriv_warp = (local_energy_sec_warp - local_energy) / da

        force_hf = -local_e_deriv
        force_hf_warp = -local_e_deriv_warp

        # compute psi derivative
        psilogderiv = (np.log(np.abs(psi_sec)) - np.log(np.abs(psi))) / da
        psilogderiv_warp = (np.log(np.abs(psi_sec_warp)) - np.log(np.abs(psi))) / da

        # Jacobian derivative
        jac_deriv = np.log(np.abs(jac))/da

        # pulay force
        energy = np.mean(local_energy)
        force_pulay = -(local_energy - energy) * 2*psilogderiv
        force_pulay_warp = -(local_energy - energy) * (2*psilogderiv_warp + jac_deriv)

        return force_hf, force_pulay, force_hf_warp, force_pulay_warp


class DMCForcesInput:

    def _retrieve_data(self, fpath):
        from numpy import array
        num_lines = sum(1 for line in open(fpath))
        data = {}
        with open(fpath, "r") as file:
            for line in tqdm.tqdm(file, total=num_lines):
                # this makes me want to cry
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

    def compute_forces(self, fpath):
        data = self._retrieve_data(fpath)

        da = data["da"].flatten()
        local_energy = data["Local energy"]
        local_energy_sec = data["Local energy (secondary)"]
        local_energy_sec_warp = data["Local energy (secondary, warp)"]

        s = data["S partial sum"]
        t = data["T partial sum"]
        ssec = data["S partial sum (secondary)"]
        tsec = data["T partial sum (secondary)"]
        ssec_warp = data["S partial sum (secondary, warp)"]
        tsec_warp = data["T partial sum (secondary, warp)"]

        weight = data["Weight"]

        psi = data["Psi"]
        psi_sec = data["Psi (secondary)"]
        psi_sec_warp = data["Psi (secondary, warp)"]
        
        jac = data["Jacobian"]
        jac_partial_sum = data["Log Jacobian partial sum"]

        # compute local e derivative
        local_e_deriv = (local_energy_sec - local_energy) / da
        local_e_deriv_warp = (local_energy_sec_warp - local_energy) / da

        force_hf = -local_e_deriv
        force_hf_warp = -local_e_deriv_warp

        # compute psi derivative
        psilogderiv = (np.log(np.abs(psi_sec)) - np.log(np.abs(psi))) / da
        psilogderiv_warp = (np.log(np.abs(psi_sec_warp)) - np.log(np.abs(psi))) / da

        # Jacobian derivative
        jac_deriv = np.log(np.abs(jac))/da

        # pulay force
        energy = np.average(local_energy, weights=weight)

        force_pulay_exact = -(local_energy - energy) * ((tsec - t)/da + (ssec - s)/da)
        force_pulay_exact_warp = -(local_energy - energy) * ((tsec_warp - t)/da + (ssec_warp - s)/da + jac_partial_sum/da)

        force_pulay_vd = -(local_energy - energy) * (2*psilogderiv + (ssec - s)/da)
        force_pulay_vd_warp = -(local_energy - energy) * (2*psilogderiv_warp + (ssec_warp - s)/da + jac_deriv)

        return force_hf, force_hf_warp, force_pulay_exact, force_pulay_exact_warp, force_pulay_vd, force_pulay_vd_warp, weight
