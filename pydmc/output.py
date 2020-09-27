from datetime import datetime
from pydmc.node_warp import *
from collections import deque
import pprint
from collections import defaultdict


class VMCLogger:

    def __init__(self, da, outfile, cutoff=lambda d: (0, 0, 0), pathak_sequence=[1e-3, 5e-3, 1e-2]):
        self._da = da
        self._cutoff = cutoff
        self._outfile = open(outfile, "w")
        self._start_time = datetime.now()
        self._pathak_sequence = pathak_sequence

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

        outdict = {
                "Configuration": x,
                "Psi": psival,
                "Gradpsi": psigrad,
                "Psi (secondary)": psisec_val,
                "Gradpsi (secondary)": psisec_grad,
                "Jacobian": jac,
                "Psi (secondary, warp)": psisec_val_warp,
                "Local energy": eloc,
                "Local energy (secondary)": eloc_prime,
                "Local energy (secondary, warp)": eloc_prime_warp,
                "Pathak regularizer": [
                    7*(d/eps)**6 - 15*(d/eps)**4 + 9*(d/eps)**2 if (d/eps) < 1 else 1.0 for eps in self._pathak_sequence
                ],
                "da": self._da
        }

        self._outfile.write(str(outdict) + "\n")


class DMCLogger:

    def __init__(self, da, outfile, lag, nwalkers, cutoff=lambda d: (0, 0, 0)):
        self._da = da
        self._cutoff = cutoff
        self._outfile = open(outfile, "w")
        self._start_time = datetime.now()
        self._counter = 0
        self._steps_per_block = lag
        self._ensemble_data = defaultdict(list)

        self._histories = {
                "grad_a S": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a T": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a S (warp)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a T (warp)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a S (no cutoff)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a T (no cutoff)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a S (warp, no cutoff)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a T (warp, no cutoff)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a sum Log Jacobian": [deque(maxlen=lag) for _ in range(nwalkers)],
        }

    def accumulate_samples(self, iwalker, walker, psi, hamiltonian, eref, tau, velocity_cutoff, num_walkers):
        if self._counter == 0:
            self._outfile.write(f"Number of walkers: {num_walkers} | Time step: {tau} | Steps per block: {self._steps_per_block}\n")

        self._counter += 1

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
        ssecnocutoff = (eref - 0.5 * (eloc_prime + eloc_prime_prev))
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
            uwarp = xwarp - xprev_warp - velocity_cutoff(vsec_warp_prev, tau)*tau

            u_nocutoff = x - xprev - vprev*tau
            uwarp_nocutoff = xwarp - xprev_warp - vsec_warp_prev*tau

            gradv_nocutoff = (vsec_prev - vprev) / self._da
            gradv_warp_nocutoff = (vsec_warp_prev - vprev) / self._da

            gradv = (velocity_cutoff(vsec_prev, tau) - velocity_cutoff(vprev, tau)) / self._da
            gradv_warp = (velocity_cutoff(vsec_warp_prev, tau) - velocity_cutoff(vprev, tau)) / self._da

            tgrad = (u @ gradv)
            tgrad_warp = (uwarp @ gradv_warp)

            tgrad_nocutoff = (u_nocutoff @ gradv_nocutoff)
            tgrad_warp_nocutoff = (uwarp_nocutoff @ gradv_warp_nocutoff)
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

    def output(self):
        weights = np.array(self._ensemble_data["Weight"])

        # for S, T and J: save a partial history of ensemble averages
        for key in self._histories:
            # push the new data into the queue for each walker
            for iwalker in range(len(self._ensemble_data[key])):
                self._histories[key][iwalker].append(self._ensemble_data[key][iwalker])
                # sum over the history
                self._ensemble_data[key][iwalker] = sum(self._histories[key][iwalker])
                # compute E_L * sum over history
                eloc = self._ensemble_data["Local energy"][iwalker]
                self._ensemble_data["E_L * " + key].append(eloc * self._ensemble_data[key][iwalker])

        # average all collected data over the ensemble of walkers
        for key, value in self._ensemble_data.items():
            self._ensemble_data[key] = np.average(value, weights=weights)
            
        self._outfile.write(str(dict(self._ensemble_data)) + "\n")

        # reset the ensemble data for the next run
        for key in self._ensemble_data:
            self._ensemble_data[key] = []


class DMCLoggerSorella:

    def __init__(self, da, outfile, lag, nwalkers, cutoff=lambda d: (0, 0, 0), sorella_eps=1e-2):
        self._da = da
        self._cutoff = cutoff
        self._outfile = open(outfile, "w")
        self._start_time = datetime.now()
        self._counter = 0
        self._steps_per_block = lag
        self._epsilon = sorella_eps
        self._ensemble_data = defaultdict(list)

        self._histories = {
                "grad_a S": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a T": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a S (warp)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a T (warp)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a S (no cutoff)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a T (no cutoff)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a S (warp, no cutoff)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a T (warp, no cutoff)": [deque(maxlen=lag) for _ in range(nwalkers)],
                "grad_a sum Log Jacobian": [deque(maxlen=lag) for _ in range(nwalkers)],
        }

    def accumulate_samples(self, iwalker, walker, psi, hamiltonian, eref, tau, velocity_cutoff, num_walkers):
        if self._counter == 0:
            self._outfile.write(f"Number of walkers: {num_walkers} | Time step: {tau} | Steps per block: {self._steps_per_block}\n")
        self._counter += 1

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
        ssecnocutoff = (eref - 0.5 * (eloc_prime + eloc_prime_prev))
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
            uwarp = xwarp - xprev_warp - velocity_cutoff(vsec_warp_prev, tau)*tau

            u_nocutoff = x - xprev - vprev*tau
            uwarp_nocutoff = xwarp - xprev_warp - vsec_warp_prev*tau

            gradv_nocutoff = (vsec_prev - vprev) / self._da
            gradv_warp_nocutoff = (vsec_warp_prev - vprev) / self._da

            gradv = (velocity_cutoff(vsec_prev, tau) - velocity_cutoff(vprev, tau)) / self._da
            gradv_warp = (velocity_cutoff(vsec_warp_prev, tau) - velocity_cutoff(vprev, tau)) / self._da

            tgrad = (u @ gradv)
            tgrad_warp = (uwarp @ gradv_warp)

            tgrad_nocutoff = (u_nocutoff @ gradv_nocutoff)
            tgrad_warp_nocutoff = (uwarp_nocutoff @ gradv_warp_nocutoff)

        # compute new distribution according to Sorella
        d = node_distance(psigrad, psival)
        deps = d if d > self._epsilon \
            else self._epsilon*(d/self._epsilon)**(d/self._epsilon)

        sorella_weight = (d / deps)**2

        # Multiply weight by Sorella weight, to recover the
        # true distribution from the modified one
        self._ensemble_data["Weight"].append(weight * sorella_weight)

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

        self._ensemble_data["Sorella weight"].append(sorella_weight)

    def output(self):
        weights = np.array(self._ensemble_data["Weight"])

        # for S, T and J: save a partial history of ensemble averages
        for key in self._histories:
            # push the new data into the queue for each walker
            for iwalker in range(len(self._ensemble_data[key])):
                self._histories[key][iwalker].append(self._ensemble_data[key][iwalker])
                # sum over the history
                self._ensemble_data[key][iwalker] = sum(self._histories[key][iwalker])
                # compute E_L * sum over history
                eloc = self._ensemble_data["Local energy"][iwalker]
                self._ensemble_data["E_L * " + key].append(eloc * self._ensemble_data[key][iwalker])

        # average all collected data over the ensemble of walkers
        for key, value in self._ensemble_data.items():
            self._ensemble_data[key] = np.average(value, weights=weights)
            
        self._outfile.write(str(self._ensemble_data) + "\n")

        # reset the ensemble data for the next run
        for key in self._ensemble_data:
            self._ensemble_data[key] = []