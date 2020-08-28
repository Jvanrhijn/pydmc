import numpy as np
import math


def node_warp(x, psival, psigrad, psisec_val, psisec_grad):
    d = node_distance(psigrad, psival)
    dprime = node_distance(psisec_grad, psisec_val)

    nprime = psisec_grad / np.linalg.norm(psisec_grad)
    n = psigrad / np.linalg.norm(psigrad)

    u, uderiv = cutoff(d)
    jac = 1 - u + np.sign(psisec_val*psival)*(u + (d - dprime)*uderiv) * (n @ nprime)
    return x + (d - dprime)*np.sign(psisec_val)*nprime*u, jac


def node_warp_fd(x, psi, psi_sec, dx=1e-7):
    warp = lambda x: _warp(x, psi, psi_sec)
    jacobian = np.zeros((len(x), len(x)))
    xbar = warp(x)
    for i in range(len(x)):
        for j in range(len(x)):
            dxx = np.zeros(len(x))
            dxx[j] = dx
            deriv = (warp(x + dxx)[i] - xbar[i]) / dx
            jacobian[i, j] = deriv
    return xbar, np.linalg.det(jacobian)


def _warp(x, psi, psi_sec):
    psival = psi(x); psigrad = psi.gradient(x)
    psiprimeval = psi_sec(x); psiprimegrad = psi_sec.gradient(x)
    d = abs(psival)/np.linalg.norm(psigrad)
    dprime = abs(psiprimeval)/np.linalg.norm(psiprimegrad)
    nprime = psiprimegrad/np.linalg.norm(psiprimegrad)
    return x + (d - dprime)*np.sign(psiprimeval)*nprime*cutoff(d)[0]



#def cutoff(d, a=0.2):
#    value = math.e*math.exp(-1/(1 - (d/a)**2)) if d < a else 0.0
#    deriv = math.e * -2*a**2*d*math.exp(-1/(1 - (d/a)**2))/(a**2 - d**2)**2 if d < a else 0.0
#    return value, deriv

def cutoff(d, a=0.1):
    if d - a <= 0:
        value = 1
        deriv = 0
    elif d - a < a:
        value = math.e*math.exp(-1/(1 - (((d-a)/a)**2))) 
        deriv = math.e * -2*(d - a)*math.exp(-1/(1 - ((d-a)/a)**2)) / (a**2 * (1 - (d-a)**2/a**2)**2)
    else:
        value = 0
        deriv = 0
    return value, deriv



#def cutoff(d, a=0.1):
#    b = a/5
#    value = 0.5 * (1 + math.tanh((a - d)/b)) if d < 4*a else 0.0
#    deriv = -1/(2*b) / math.cosh((a - d)/b)**2 if d < 4*a else 0.0
#    return value, deriv


def node_distance(psigrad, psi):
    return abs(psi) / np.linalg.norm(psigrad)