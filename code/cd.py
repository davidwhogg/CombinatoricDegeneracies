"""
This file is part of the CombinatoricDegeneracies project.
Copyright 2016 David W. Hogg & Dan Foreman-Mackey.
"""

import itertools as it
import numpy as np

def get_log_likelihood(K, D):
    """
    Build a log likelihood function for a problem with `K` words, each
    of which has `D` adjustable parameters.  The output function will
    take as input a numpy array with `K*D` elements.
    """
    assert int(K) > 0
    assert int(D) > 0
    if K > 8:
        print("get_log_likelihood(): seriously? you want K = ", K)
        assert False

    # create K D-space Gaussians
    amps = np.ones(K)
    means = np.zeros((K, D))
    ivars = np.zeros((K, D, D))
    ds = np.arange(D)
    ks = np.arange(K)
    for d in ds:
        means[ks, d] = np.sin(np.pi * ks * (d + 1.) / (2. * K))
    for k in ks:
        ivars[k] = 2. ** (8. - k) * np.eye(D)
    print(amps, means, ivars)

    # create mixture of K! Gaussians (OMG crap!)
    Kfac = sum([1 for q in it.permutations(ks)]) # HACKETY HACK
    KD = K * D
    bigamps = np.zeros(Kfac)
    bigmeans = np.zeros((Kfac, KD))
    bigivars = np.zeros((Kfac, KD, KD))
    for i, p in enumerate(it.permutations(ks)):
        bigamps[i] = np.prod(amps)
        for k in ks:
            bigmeans[i, k * D : k * D + D] = means[p[k]]
            bigivars[i, k * D : k * D + D, k * D : k * D + D] = ivars[p[k]]
    print(bigamps, bigmeans, bigivars)

    def ln_like(pars):
        # return mixture_of_gaussians(bigamps, bigmeans, bigivars)
        return 0.
    return ln_like

if __name__ == "__main__":
    ln_like = get_log_likelihood(3, 1)
