import numpy as np


def node_warp(x, psival, psigrad, psisec_val, psisec_grad):
    d = node_distance(psigrad, psival)
    dprime = node_distance(psisec_grad, psisec_val)

    nprime = psisec_grad / np.linalg.norm(psisec_grad)
    n = psigrad / np.linalg.norm(psigrad)

    u, uderiv = cutoff(d)
    jac = 1 - u + np.sign(psisec_val*psival)*(u + (d - dprime)*uderiv) * (n @ nprime)
    return x + (d - dprime)*np.sign(psisec_val)*nprime*u, jac


def cutoff(d, a=0.1):
    b = a/5
    value = 0.5 * (1 + np.tanh((a - d)/b))
    deriv = -1/(2*b) / np.cosh((a - d)/b)**2
    #return 0, 0
    return value, deriv


def node_distance(psigrad, psi):
    return abs(psi) / np.linalg.norm(psigrad)