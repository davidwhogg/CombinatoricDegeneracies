"""
This file is part of the CombinatoricDegeneracies project.
Copyright 2016 David W. Hogg & Dan Foreman-Mackey.

# to-do list:
- Make multiply function that can multiply the likelihood function by a prior.
- Figure out how to visualize things.
- Write sampling code.
"""

import itertools as it
import numpy as np
import pylab as plt

def factorial(N):
    if N == 0:
        return 1
    assert N > 0
    return np.prod(range(1, N+1))

def choose(N, M):
    """
    ## bugs:
    - bad implementation (will fail at large inputs)
    """
    return factorial(N) / factorial(M) / factorial(N-M)

class mixture_of_gaussians:
    """
    ## bugs:
    - May not work at `K==1` or `D==1`.
    """

    def __init__(self, amps, means, ivars):
        self.K = len(amps)
        assert amps.shape == (self.K,)
        assert np.all(amps > 0.)
        self.amps = amps
        KK, D = means.shape
        assert KK == self.K
        self.D = D
        self.means = means
        assert ivars.shape == (self.K, self.D, self.D)
        for ivar in ivars:
            assert np.all(ivar.T == ivar)
        self.ivars = ivars
        self.logdets = np.zeros(self.K)
        for ivar in ivars:
            slogdet = np.linalg.slogdet(ivar)
            print(slogdet)

    def value(self, x):
        assert x.shape == self.means[0].shape
        val = 0.
        for k in range(self.K):
            delta = x - self.means[k]
            val += self.amps[k] * np.exp(-0.5 * np.dot(delta, np.dot(self.ivars[k], delta)) + 0.5 * self.logdets[k])
        return val

    def __call__(self, x):
        return np.log(self.value(x))

def get_log_likelihood(M, K, D, ivar_scale=256., ndof=None):
    """
    Build a log likelihood function for a problem with `K` pigeons,
    each of which gets put in one of `M` holes, each of which has `D`
    adjustable parameters.  The output function will take as input a
    numpy array with `K*D` elements.

    ## notes:
    - Makes amplitudes from a flat-in-log pdf.
    - Makes means from a unit-variance Gaussian.
    - Makes inverse variances from a mean of outer products of things.

    ## bugs:
    - Should take random state as input.
    - Magic numbers and decisions galore.
    - Doesn't work yet.
    """
    assert int(K) > 0
    assert int(D) > 0
    assert M > K

    # create M D-space Gaussians
    amps = np.exp(np.random.uniform(size=M)) # MAGIC decision
    means = np.random.normal(size=(M, D)) # more MAGIC
    ivars = np.zeros((M, D, D))
    if ndof is None:
        ndof = D + 2 # MAGIC
    for m in range(M):
        vecs = np.random.normal(size=(ndof, D)) # more MAGIC
        ivars[m] = ivar_scale * np.mean(vecs[:, :, None] * vecs[:, None, :], axis=0)
    print(amps, means, ivars)

    # create mixture of M-choose-K times K! Gaussians (OMG!)
    Kfac = factorial(K)
    McKKfac = choose(M, K) * Kfac
    KD = K * D
    bigamps = np.zeros(McKKfac)
    bigmeans = np.zeros((McKKfac, KD))
    bigivars = np.zeros((McKKfac, KD, KD))
    for i, p in enumerate(it.permutations(range(M), K)):
        bigamps[i] = 1.
        for k in range(K):
            bigamps[i] *= amps[p[k]]
            bigmeans[i, k * D : k * D + D] = means[p[k]]
            bigivars[i, k * D : k * D + D, k * D : k * D + D] = ivars[p[k]]
    bigamps /= McKKfac

    return mixture_of_gaussians(bigamps, bigmeans, bigivars)

if __name__ == "__main__":
    ln_like = get_log_likelihood(7, 5, 3)
    plt.clf()
    xis = np.arange(-2., 2., 0.01)
    xs = np.zeros((len(xis), 5 * 3))
    xs[:,0] = xis
    ln_Ls = np.array([ln_like(x) for x in xs])
    plt.plot(xis, np.exp(ln_Ls - np.max(ln_Ls)), "k-")
    plt.savefig("cd.png")
