import copy
import unittest


import numpy as np
import matplotlib.pyplot as plt


def gradient_fd(fun, x, dx=1e-5):
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


def laplacian_fd(fun, x, dx=1e-5):
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
