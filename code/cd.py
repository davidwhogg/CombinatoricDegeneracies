"""
This file is part of the CombinatoricDegeneracies project.
Copyright 2016 David W. Hogg & Dan Foreman-Mackey.
"""

import itertools as it
import numpy as np

def factorial(N):
    if N == 0:
        return 1
    assert N > 0
    return np.prod(range(1, N+1))

def choose(N, M):
    """
    bad implementation
    """
    return factorial(N) / factorial(M) / factorial(N-M)

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
    print(bigamps)

    def ln_like(pars):
        # return mixture_of_gaussians(bigamps, bigmeans, bigivars)
        return 0.
    return ln_like

if __name__ == "__main__":
    ln_like = get_log_likelihood(7, 5, 3)
