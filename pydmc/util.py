import math
import copy
import unittest


import numpy as np
import matplotlib.pyplot as plt


def gradient_fd(fun, x, dx=1e-3):
    """
    Compute the gradient of `fun` using finite differences, evaluated at `x`.
    Equal increments in all directions are assumed.

    Parameters
    ----------
    fun: function
        Takes a single vector parameter `x`.
    x: np.ndarray
        Position on which to sample gradient of fun.
    """
    grad = np.zeros(x.shape)
    dimension = len(x)
    for i in range(dimension):
        dxx = np.zeros(dimension)
        dxx[i] = dx
        grad[i] = (fun(x + dxx) - fun(x - dxx)) / (2*dx)
    return grad


def laplacian_fd(fun, x, dx=1e-3):
    """
    Compute the Laplacian of `fun` using finite differences, evaluated at `x`.

    Parameters
    ----------
    fun: function
        Takes a single vector parameter `x`
    x: np.ndarray
        Position on which to sample gradient of fun.
    """
    lapl = 0.0
    dimension = len(x)
    for i in range(dimension):
        dxx = np.zeros(dimension)
        dxx[i] = dx
        lapl += (fun(x + dxx) - 2*fun(x) + fun(x - dxx)) / dx**2
    return lapl


def velocity_cutoff_umrigar(v, tau, a=1):
    vnorm = np.linalg.norm(v)
    return v * (-1 + math.sqrt(1 + 2*a*vnorm**2*tau)) / (a*vnorm**2*tau)


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def bin_samples(samples, bin_size=100):
    """Bin samples for blocking analysis of data traces"""
    nsamples = len(samples)
    blocks = np.array_split(samples, nsamples//bin_size)
    block_avgs = np.array([np.mean(b) for b in blocks])
    return block_avgs


def block_error(data, block_size, weights=None):
    num_blocks = len(data) // block_size
    blocks = np.array_split(data, num_blocks)
    bmeans = np.array([np.mean(b) for b in blocks])
    mean = np.mean(bmeans)
    meansq = np.mean(bmeans**2)
    return np.sqrt((meansq - mean**2) / (len(blocks) -1 ))


def flatten(l):
    return [item for sublist in l for item in sublist]


def munu(x, a):
    x1, x2 = x[0], x[1]
    mu = np.real(np.arccosh((x1 + 1j*x2)/a))
    nu = np.imag(np.arccosh((x1 + 1j*x2)/a))
    return np.array([mu, nu])




class TestFunctions(unittest.TestCase):

    def test_gradient(self):
        f = lambda x: (x @ x)
        grad = lambda x: 2*x
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(grad(x), gradient_fd(f, x), decimal=4)

    def test_laplacian(self):
        f = lambda x: (x @ x)
        lapl = lambda x: 2*len(x)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(lapl(x), laplacian_fd(f, x), decimal=4)



if __name__ == '__main__':
    unittest.main()