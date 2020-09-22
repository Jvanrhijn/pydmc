from datetime import datetime
from pydmc.node_warp import *
from collections import deque


class VMCLogger:

    def __init__(self, da, outfile, cutoff=lambda d: (0, 0, 0)):
        self._da = da
        self._cutoff = cutoff
        self._outfile = open(outfile, "w")
        self._start_time = datetime.now()

    def accumulate_samples(self, conf, psi, hamiltonian, tau):
        x = conf
        psival = psi(x)
        psigrad = psi.gradient(x)

        psi_sec = psi.deform(self._da)

        psisec_val = psi_sec(x)
        psisec_grad = psi_sec.gradient(x)

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
                "da": self._da
        }

        self._outfile.write(str(outdict) + "\n")


class DMCLogger:

    def __init__(self, da, outfile, lag, cutoff=lambda d: (0, 0, 0)):
        self._da = da
        self._cutoff = cutoff
        self._outfile = open(outfile, "w")
        self._start_time = datetime.now()

        self._s_history = deque(maxlen=lag)
        self._t_history = deque(maxlen=lag)
        self._ssec_history = deque(maxlen=lag)
        self._tsec_history = deque(maxlen=lag)
        self._ssec_warp_history = deque(maxlen=lag)
        self._tsec_warp_history = deque(maxlen=lag)
        self._jac_history = deque(maxlen=lag)

        self._counter = 0

        self._ensemble_data = {
                "Weight": [],
                "Psi": [],
                "Psi (secondary)": [],
                "Log Jacobian": [],
                "Sum Log Jacobian": [],
                "Psi (secondary, warp)": [],
                "Local energy": [],
                "Local energy (secondary)": [],
                "Local energy (secondary, warp)": [],
                "grad_a E_L": [],
                "grad_a E_L (warp)": [],
                "grad_a Log Psi": [],
                "grad_a Log Psi (warp)": [],
                "grad_a Log Jacobian": [],
                "E_L * grad_a Log Jacobian": [],
                "E_L * grad_a Log Psi": [],
                "E_L * grad_a Log Psi (warp)": [],
                "S": [],
                "T": [],
                "S (secondary)": [],
                "T (secondary)": [],
                "S (secondary, warp)": [],
                "T (secondary, warp)": [],
        }

        self._histories = {
                "S": deque(maxlen=lag),
                "T": deque(maxlen=lag),
                "S (secondary)": deque(maxlen=lag),
                "T (secondary)": deque(maxlen=lag),
                "S (secondary, warp)": deque(maxlen=lag),
                "T (secondary, warp)": deque(maxlen=lag),
                "Sum Log Jacobian": deque(maxlen=lag),
        }

    def accumulate_samples(self, iwalker, walker, psi, hamiltonian, eref, tau, velocity_cutoff, num_walkers):
        if self._counter == 0:
            self._outfile.write(f"Number of walkers: {num_walkers} | Time step: {tau} | da: {self._da}\n")
        self._counter += 1

        x = walker.configuration
        xprev = walker.previous_configuration
        weight = walker.weight 

        psival = psi(x)
        psigrad = psi.gradient(x)
        v = psigrad / psival

        psival_prev = psi(xprev)
        psigrad_prev = psi.gradient(xprev)
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
            t = 0
            tsec = 0
            tsec_warp = 0
        else:        
            t = -np.linalg.norm(x - xprev - velocity_cutoff(vprev, tau)*tau)**2 / (2*tau)
            tsec = -np.linalg.norm(x - xprev - velocity_cutoff(vsec_prev, tau)*tau)**2 / (2*tau)
            tsec_warp = -np.linalg.norm(xwarp - xprev_warp - velocity_cutoff(vsec_warp_prev, tau)*tau)**2 / (2*tau)

        # increment the partial sums of t and s
        self._s_history.append(s)
        self._t_history.append(t)
        self._ssec_history.append(ssec)
        self._tsec_history.append(tsec)
        self._ssec_warp_history.append(ssec_warp)
        self._tsec_warp_history.append(tsec_warp)
        self._jac_history.append(math.log(abs(jac)))

        #outdict = {
        self._ensemble_data["Weight"].append(weight)

        self._ensemble_data["Psi"].append(psival)
        self._ensemble_data["Psi (secondary)"].append(psisec_val)
        self._ensemble_data["Psi (secondary, warp)"].append(psisec_val_warp)

        self._ensemble_data["Log Jacobian"].append(math.log(abs(jac)))
        self._ensemble_data["Sum Log Jacobian"].append(math.log(abs(jac)))

        self._ensemble_data["Local energy"].append(eloc)
        self._ensemble_data["Local energy (secondary)"].append(eloc_prime)
        self._ensemble_data["Local energy (secondary, warp)"].append(eloc_prime_warp)

        self._ensemble_data["S"].append(s)
        self._ensemble_data["T"].append(t) 
        self._ensemble_data["S (secondary)"].append(ssec)
        self._ensemble_data["T (secondary)"].append(tsec)
        self._ensemble_data["S (secondary, warp)"].append(ssec_warp)
        self._ensemble_data["T (secondary, warp)"].append(tsec_warp)

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
        # average all collected data over the ensemble
        for key, value in self._ensemble_data.items():
            self._ensemble_data[key] = np.average(value, weights=weights, axis=0)

        # for S, T and J: save a partial history of ensemble averages
        for key in self._histories:
            self._histories[key].append(self._ensemble_data[key])
            # calculate the sum of the history for output
            self._ensemble_data[key] = sum(self._histories[key])

        self._outfile.write(str(self._ensemble_data) + "\n")

        # reset the ensemble data for the next run
        for key in self._ensemble_data:
            self._ensemble_data[key] = []
