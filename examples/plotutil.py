import matplotlib.pyplot as plt
import numpy as np
import scipy.special as spec
from pydmc import *


def plot_force_data_trace(flhf, flpulay, flhf_warp, flpulay_warp, bin_size=1):
    bs = bin_size
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
    axes[0].plot(bin_samples(flhf, bin_size=bs), label="No warp")
    axes[0].plot(bin_samples(flhf_warp, bin_size=bs), label="warp")

    axes[1].plot(bin_samples(flpulay, bin_size=bs), label="No warp")
    axes[1].plot(bin_samples(flpulay_warp, bin_size=bs), label="warp")

    axes[2].plot(bin_samples((flhf + flpulay), bin_size=bs), label="No warp")
    axes[2].plot(bin_samples((flhf_warp + flpulay_warp), bin_size=bs), label="warp")
    titles = ["Hellmann-Feynman Force", "Pulay Force", "Total Force"]
    for title, ax in zip(titles, axes):
        ax.legend(); ax.grid()
        ax.set_title(title)


# function to compute error bar given a slice of local force
def error_over_time(data, steps_per_block, num_points, weights=None):
    if weights is None:
        weights = np.ones(data.shape)
    partition_size = len(data) // num_points
    errs = np.array([block_error(data[:(i+1)*partition_size], block_size=steps_per_block) for i in range(num_points)])
    means = np.array([np.average(data[:(i+1)*partition_size], weights=weights[:(i+1)*partition_size]) for i in range(num_points)])
    return means, errs


def plot_error_over_time(flhf, flpulay, flhf_warp, flpulay_warp, npoints, steps_per_block, weights=None):
    ns = np.linspace(1, len(flhf)//steps_per_block, npoints)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey=False)
    for i, (f, fwarp) in enumerate([(flhf, flhf_warp), (flpulay, flpulay_warp), (flhf + flpulay, flhf_warp + flpulay_warp)]):
        means, errs = error_over_time(f, steps_per_block, npoints, weights=weights)
        means_warp, errs_warp = error_over_time(fwarp, steps_per_block, npoints, weights=weights)
        axes[0, i].errorbar(ns, means, yerr=errs, marker='o', label="Not warped")
        axes[1, i].plot(ns, errs, marker='o', label="Not warped")
        axes[0, i].errorbar(ns, means_warp, yerr=errs_warp, marker='o', label="Warped")
        axes[1, i].plot(ns, errs_warp, marker='o', label="Warped")
        axes[0, i].legend(); axes[0, i].grid()
        axes[1, i].legend(); axes[1, i].grid()
    return axes



q0 = 0.8253525490491695

def mathieu_modcem1(m, q, x):
    return spec.mathieu_modcem1(m, q, x)

def mathieu_cem(m, q, x):
    # Angular mathieu function in scipy takes input
    # in degrees
    return spec.mathieu_cem(m, q, x*180/np.pi)

def ground_state(mu, nu):
    if mu > 1:
        return 0
    return mathieu_modcem1(0, q0, mu)[0]*mathieu_cem(0, q0, nu)[0]


class GuidingWaveFunction(WaveFunction, GeometryParametrized):

    def __init__(self, a):
        self._a = a
        self._alpha = self._a[0]*math.cosh(1)
        self._beta = self._a[0]*math.sinh(1)

    def __call__(self, x):
        x, y = x
        mu = np.real(np.arccosh((x + 1j*y)/self._a[0]))
        nu = np.imag(np.arccosh((x + 1j*y)/self._a[0]))
        return -x*ground_state(mu, nu)

    def gradient(self, xx):
        return gradient_fd(self, xx, dx=1e-4)
    
    def laplacian(self, xx):
        return laplacian_fd(self, xx, dx=1e-4)

    def geometry_parameters(self):
        return self._a

    def deform(self, parameter_change):
        return GuidingWaveFunction(self._a + parameter_change)

    def geometry_gradient(self, x, da=1e-4):
        psisec = self.deform(np.array([da]))
        return np.array((psisec(x) - self(x))/da)


class ParabolicWaveFunction:

    def __init__(self, a):
        self._a = a
        self._alpha = self._a*math.cosh(1)
        self._beta = self._a*math.sinh(1)
        self._ellipse_pars = np.array([self._alpha, self._beta])

    def __call__(self, x):
        r = np.sum((x / self._ellipse_pars)**2)
        if r > 1:
            return 0
        return 1 - r

    def gradient(self, xx):
        return -2 * xx / self._ellipse_pars**2
    
    def laplacian(self, xx):
        return -2 * (1/self._ellipse_pars**2).sum()

    def geometry_parameters(self):
        return self._a

    def deform(self, parameter_change):
        return ParabolicWaveFunction(self._a + parameter_change)

    def geometry_gradient(self, x):
        x, y = x
        return 2*(x**2/self._alpha**3*math.cosh(1) \
                        + y**2/self._beta**3*math.sinh(1))