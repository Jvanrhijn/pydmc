from datetime import datetime
from pydmc.node_warp import *
from pydmc.util import circularlist
from collections import deque
import pprint
from collections import defaultdict
import h5py



class HDF5Logger:

    def __init__(self, file):
        self._outfile = h5py.File(file, "w")

    @staticmethod
    def resize(name, node):
        if isinstance(node, h5py.Dataset) and len(node.shape) == 2:
            node.resize((node.shape[0]+1, node.shape[1]))
        else:
            pass

    def close_file(self):
        self._outfile.close()


class VMCLogger(HDF5Logger):

    def __init__(self, da, outfile, cutoff=lambda d: (0, 0, 0), pathak_sequence=[1e-3, 5e-3, 1e-2]):
        super().__init__(outfile)
        self._da = da
        self._cutoff = cutoff
        self._start_time = datetime.now()
        self._pathak_sequence = pathak_sequence
        self._block_data = defaultdict(list)

        scalar_keys = [
            "grad_a log Psi", 
            "grad_a log Psi (warp)", 
            "grad_a log Jacobian", 
            "E_L * grad_a log Psi", 
            "E_L * grad_a log Psi (warp)", 
            "E_L * grad_a log Jacobian",
            "Local energy", 
            "grad_a E_L", 
            "grad_a E_L (warp)"
        ]
        for key in scalar_keys:
            self._outfile.create_dataset(key, (1, 1), maxshape=(None, 1), chunks=(1, 1))

    def accumulate_samples(self, conf, psi, hamiltonian, tau):
        x = conf
        psival = psi(x)
        psigrad = psi.gradient(x)

        psi_sec = psi.deform(self._da)

        psisec_val = psi_sec(x)
        psisec_grad = psi_sec.gradient(x)

        d = node_distance(psigrad, psival)

        xwarp, jac, j \
            = node_warp(x, psival, psigrad, psisec_val, psisec_grad, cutoff=self._cutoff)
        
        psisec_val_warp = psi_sec(xwarp)
        
        eloc = hamiltonian(psi, x) / psival
        eloc_prime = hamiltonian(psi_sec, x) / psisec_val
        eloc_prime_warp = hamiltonian(psi_sec, xwarp) / psisec_val_warp

        el_grad = (eloc_prime - eloc) / self._da
        el_grad_warp = (eloc_prime_warp - eloc) / self._da

        logpsi_grada = (math.log(abs(psisec_val)) - math.log(abs(psival))) / self._da
        logpsi_grada_warp = (math.log(abs(psisec_val_warp)) - math.log(abs(psival))) / self._da

        block_data = {
                "grad_a log Psi": logpsi_grada,
                "grad_a log Psi (warp)": logpsi_grada_warp,
                "grad_a log Jacobian": math.log(abs(jac))/self._da,
                "E_L * grad_a log Psi": eloc*logpsi_grada,
                "E_L * grad_a log Psi (warp)": eloc*logpsi_grada_warp,
                "E_L * grad_a log Jacobian": eloc*math.log(abs(jac))/self._da,
                "Local energy": eloc,
                "grad_a E_L": el_grad,
                "grad_a E_L (warp)": el_grad_warp,
                #"Pathak regularizer": [
                    #7*(d/eps)**6 - 15*(d/eps)**4 + 9*(d/eps)**2 if (d/eps) < 1 else 1.0 for eps in self._pathak_sequence
                #],
        }

        for key, value in block_data.items():
            self._block_data[key].append(value)

    def output(self):
        # resize data sets
        self._outfile.visititems(self.resize)

        # update 
        for key, value in self._block_data.items():
            self._outfile[key][-1] = np.mean(value)

        self._block_data = defaultdict(list)


class DMCLogger(HDF5Logger):

    def __init__(self, da, outfile, nwalkers, cutoff=lambda d: (0, 0, 0)):
        super().__init__(outfile)
        self._da = da
        self._cutoff = cutoff
        self._counter = 0
        self._ensemble_data = defaultdict(list)
        self._block_data = defaultdict(list)

        scalar_keys = [
            "Weight",
            "Psi",
            "Local energy",
            "grad_a S",
            "grad_a T",
            "grad_a S (warp)",
            "grad_a T (warp)",
            "grad_a S (no cutoff)",
            "grad_a T (no cutoff)",
            "grad_a S (warp, no cutoff)",
            "grad_a T (warp, no cutoff)",
            "grad_a sum Log Jacobian",
            "grad_a E_L",
            "grad_a E_L (warp)",
            "grad_a Log Psi",
            "grad_a Log Psi (warp)",
            "E_L * grad_a Log Psi",
            "E_L * grad_a Log Psi (warp)",
            "grad_a Log Jacobian",
            "E_L * grad_a Log Jacobian"
        ]

        history_keys = [
                "grad_a S",
                "grad_a T",
                "grad_a S (warp)",
                "grad_a T (warp)",
                "grad_a S (no cutoff)",
                "grad_a T (no cutoff)",
                "grad_a S (warp, no cutoff)",
                "grad_a T (warp, no cutoff)",
                "grad_a sum Log Jacobian",
        ]

        for key in scalar_keys:
            self._outfile.create_dataset(key, (1, 1), maxshape=(None, 1), chunks=(1, 1))

        for key in history_keys:
            self._outfile.create_dataset("E_L * " + key, (1, 1), maxshape=(None, 1), chunks=(1, 1))


    def accumulate_samples(self, iwalker, walker, psi, hamiltonian, eref, tau, velocity_cutoff, num_walkers):
        x = walker.configuration
        xprev = walker.previous_configuration
        weight = walker.weight 

        psival = walker.value
        psigrad = walker.gradient
        v = psigrad / psival


        psival_prev = walker.previous_value
        psigrad_prev = walker.previous_gradient
        vprev = psigrad_prev / psival_prev

        psi_sec = psi.deform(self._da)

        psisec_val = psi_sec(x)
        psisec_grad = psi_sec.gradient(x)
        vsec = psisec_grad / psisec_val

        psisec_val_prev = psi_sec(xprev)
        psisec_grad_prev = psi_sec.gradient(xprev)
        vsec_prev = psisec_grad_prev / psisec_val_prev

        xwarp, jac, _ \
            = node_warp(x, psival, psigrad, psisec_val, psisec_grad, cutoff=self._cutoff)
        xprev_warp, _, _ \
            = node_warp(xprev, psival_prev, psigrad_prev, psisec_val_prev, psisec_grad_prev, cutoff=self._cutoff)

        psisec_val_warp = psi_sec(xwarp)
        psisec_val_prev_warp = psi_sec(xprev_warp)
        vsec_warp_prev = psi_sec.gradient(xprev_warp) / psisec_val_prev_warp
        vsec_warp = psi_sec.gradient(xwarp) / psisec_val_warp

        eloc = hamiltonian(psi, x) / psival
        eloc_prime = hamiltonian(psi_sec, x) / psisec_val
        eloc_prime_warp = hamiltonian(psi_sec, xwarp) / psisec_val_warp
        eloc_prime_prev = hamiltonian(psi_sec, xprev) / psisec_val_prev
        eloc_prev = hamiltonian(psi, xprev) / psi(xprev)
        eloc_prime_prev_warp = hamiltonian(psi_sec, xprev_warp) / psi_sec(xprev_warp)

        snocutoff = (eref - 0.5 * (eloc_prev + eloc)) * tau
        ssecnocutoff = (eref - 0.5 * (eloc_prime + eloc_prime_prev)) * tau
        ssecnocutoff_warp = (eref - 0.5 * (eloc_prime_warp + eloc_prime_prev_warp)) * tau

        vbar = velocity_cutoff(v, tau)
        s = (eref - 0.5 * (eloc_prev + eloc)) * tau \
            * np.linalg.norm(vbar)/np.linalg.norm(v)

        vbar_sec = velocity_cutoff(vsec, tau)
        ssec = (eref - 0.5 * (eloc_prime + eloc_prime_prev)) * tau \
            * np.linalg.norm(vbar_sec)/np.linalg.norm(vsec)

        vbar_sec_warp = velocity_cutoff(vsec_warp, tau)
        ssec_warp = (eref - 0.5 * (eloc_prime_warp + eloc_prime_prev_warp)) * tau \
            * np.linalg.norm(vbar_sec_warp)/np.linalg.norm(vsec_warp)

        if np.all(x == xprev):
            tgrad = 0
            tgrad_warp = 0

            tgrad_nocutoff = 0
            tgrad_warp_nocutoff = 0
        else:        
            u = x - xprev - velocity_cutoff(vprev, tau)*tau
            u_nocutoff = x - xprev - vprev*tau

            gradv_nocutoff = (vsec_prev - vprev) / self._da
            gradv_warp_nocutoff = (vsec_warp_prev - vprev) / self._da

            gradv = (velocity_cutoff(vsec_prev, tau) - velocity_cutoff(vprev, tau)) / self._da
            gradv_warp = (velocity_cutoff(vsec_warp_prev, tau) - velocity_cutoff(vprev, tau)) / self._da

            tgrad = u @ gradv
            tgrad_warp = u @ gradv_warp

            tgrad_nocutoff = u_nocutoff @ gradv_nocutoff
            tgrad_warp_nocutoff = u_nocutoff @ gradv_warp_nocutoff
#
        self._ensemble_data["Weight"].append(weight)

        self._ensemble_data["Psi"].append(psival)

        self._ensemble_data["Local energy"].append(eloc)

        self._ensemble_data["grad_a S"].append((ssec - s) / self._da)
        self._ensemble_data["grad_a T"].append(tgrad)

        self._ensemble_data["grad_a S (warp)"].append((ssec_warp - s) / self._da)
        self._ensemble_data["grad_a T (warp)"].append(tgrad_warp)

        self._ensemble_data["grad_a S (no cutoff)"].append((ssecnocutoff - snocutoff) / self._da)
        self._ensemble_data["grad_a T (no cutoff)"].append(tgrad_nocutoff)

        self._ensemble_data["grad_a S (warp, no cutoff)"].append((ssecnocutoff_warp - snocutoff) / self._da)
        self._ensemble_data["grad_a T (warp, no cutoff)"].append(tgrad_warp_nocutoff)

        self._ensemble_data["grad_a sum Log Jacobian"].append(math.log(abs(jac))/ self._da) 
        self._ensemble_data["grad_a E_L"].append((eloc_prime - eloc) / self._da)
        self._ensemble_data["grad_a E_L (warp)"].append((eloc_prime_warp - eloc) / self._da)

        self._ensemble_data["grad_a Log Psi"].append((math.log(abs(psisec_val)) - math.log(abs(psival))) / self._da)
        self._ensemble_data["grad_a Log Psi (warp)"].append((math.log(abs(psisec_val_warp)) - math.log(abs(psival))) / self._da)

        self._ensemble_data["E_L * grad_a Log Psi"].append(eloc * (math.log(abs(psisec_val)) - math.log(abs(psival))) / self._da)
        self._ensemble_data["E_L * grad_a Log Psi (warp)"].append(eloc * (math.log(abs(psisec_val_warp)) - math.log(abs(psival))) / self._da)

        self._ensemble_data["grad_a Log Jacobian"].append(math.log(abs(jac)) / self._da)
        self._ensemble_data["E_L * grad_a Log Jacobian"].append(eloc * math.log(abs(jac)) / self._da)

    def average_ensemble(self, walkers):
        # for S, T and J: save a partial history in the walkers
        for key in walkers[0].histories:
            for iwalker in range(len(walkers)):
                # grab the history of this walker
                history = walkers[iwalker].histories[key]

                # add new data to walker history
                history.append(self._ensemble_data[key][iwalker])

                # sum over the history of this walker
                history_sum = sum(history)

                # Plug history sums into ensemble
                self._ensemble_data[key][iwalker] = history_sum

                # compute E_L * sum over history and plug into ensemble
                eloc = self._ensemble_data["Local energy"][iwalker]
                self._ensemble_data["E_L * " + key].append(eloc * history_sum)

        # average all collected data over the ensemble of walkers
        for key, value in self._ensemble_data.items():
            self._ensemble_data[key] = np.mean(value)
            
        # write the ensemble averages to block storage
        for key in self._ensemble_data:
            self._block_data[key].append(self._ensemble_data[key])

        # reset the ensemble data for the next run
        self._ensemble_data = defaultdict(list)

    def output(self):
        # resize data sets
        self._outfile.visititems(self.resize)

        # update output file with block averages
        for key, value in self._block_data.items():
            self._outfile[key][-1] = np.mean(value)

        # reset block data for next pass
        self._block_data = defaultdict(list)