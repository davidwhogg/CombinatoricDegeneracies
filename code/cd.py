"""
This file is part of the CombinatoricDegeneracies project.
Copyright 2016 David W. Hogg & Dan Foreman-Mackey.

# to-do list:
- Write code to return a marginalized likelihood, given a prior.
- Write exact-sampling code.

# notes / bugs:
- The zero covariance across modes (block-diagonality) is baked-in.
"""

import itertools as it
import numpy as np
from scipy.misc import logsumexp

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
    ## notes:
    - Redundantly, this requires *both* `vars` and `ivars`. Why? Good reasons!

    ## bugs:
    - May not work at `K==1` or `D==1`.
    """

    def __init__(self, amps, means, vars, ivars):
        self.K = len(amps)
        assert amps.shape == (self.K,)
        assert np.all(amps >= 0.)
        self.amps = amps
        KK, D = means.shape
        assert KK == self.K
        self.D = D
        self.means = means
        assert vars.shape == (self.K, self.D, self.D)
        assert ivars.shape == (self.K, self.D, self.D)
        for var, ivar in zip(vars, ivars):
            assert np.all(var.T == var)
            assert np.all(ivar.T == ivar)
            assert np.allclose(np.dot(var, ivar), np.eye(self.D))
        self.vars = vars
        self.ivars = ivars
        self.logdets = np.zeros(self.K)
        for k,var in enumerate(vars):
            s, logdet = np.linalg.slogdet(var)
            assert s > 0.
            self.logdets[k] = logdet

    def __mul__(self, other):
        """
        multiply one by another!

        ## bugs:
        - Only partially tested.
        """
        assert self.D == other.D
        newK = self.K * other.K
        newamps = np.zeros(newK)
        newmeans = np.zeros((newK, self.D))
        newvars = np.zeros((newK, self.D, self.D))
        newivars = np.zeros((newK, self.D, self.D))
        for k,(sk,ok) in enumerate(it.product(range(self.K), range(other.K))):
            # following https://www.cs.nyu.edu/~roweis/notes/gaussid.pdf
            newivars[k] = self.ivars[sk] + other.ivars[ok]
            newvars[k] = np.linalg.inv(newivars[k])
            newvars[k] = 0.5 * (newvars[k] + newvars[k].T) # symmetrize
            newmeans[k] = np.dot(newvars[k], (np.dot(self.ivars[sk], self.means[sk]) +
                                              np.dot(other.ivars[ok], other.means[ok])))
            s, newlogdet = np.linalg.slogdet(newvars[k])
            assert s > 0.
            newlogamp = np.log(self.amps[sk]) + np.log(other.amps[ok]) - 0.5 * self.D * np.log(np.pi) \
                + 0.5 * newlogdet - 0.5 * self.logdets[sk] - 0.5 * other.logdets[ok] \
                - 0.5 * np.dot(self.means[sk],  np.dot(self.ivars[sk],  self.means[sk])) \
                - 0.5 * np.dot(other.means[ok], np.dot(other.ivars[ok], other.means[ok])) \
                + 0.5 * np.dot(newmeans[k], np.dot(newivars[k], newmeans[k]))
            newamps[k] = np.exp(newlogamp)
        return mixture_of_gaussians(newamps, newmeans, newvars, newivars)

    def draw_sample(self):
        k = np.random.randint(self.K)
        return np.random.multivariate_normal(self.means[k], self.vars[k])

    def log_value(self, x):
        """
        log as in ln
        """
        assert x.shape == self.means[0].shape
        vals = np.log(self.amps) - 0.5 * self.D * np.log(np.pi) - 0.5 * self.logdets
        for k in range(self.K):
            delta = x - self.means[k]
            vals[k] += -0.5 * np.dot(delta, np.dot(self.ivars[k], delta)) # chi-squared term
        return logsumexp(vals)

    def log_marginalized_value(self, d, xd):
        """
        Same as `log_value` but plotting the one-dimensional function,
        marginalizing out everything except dimension `d`.

        ## notes:
        - Check out `var` not `ivar`.
        """
        vals = np.log(self.amps) - np.log(np.pi) - 0.5 * np.log(self.vars[:, d, d])
        for k in range(self.K):
            delta = xd - self.means[k, d]
            vals[k] += -0.5 * delta * delta / self.vars[k, d, d]
        return logsumexp(vals)

    def log_fully_marginalized_value(self):
        return np.log(np.sum(self.amps))

    def __call__(self, x, d=None):
        if d is None:
            return self.log_value(x)
        return self.log_marginalized_value(d, x)

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
    vars = np.zeros((M, D, D))
    ivars = np.zeros((M, D, D))
    if ndof is None:
        ndof = D + 2 # MAGIC
    for m in range(M):
        vecs = np.random.normal(size=(ndof, D)) # more MAGIC
        vars[m] = (1. / ivar_scale) * np.mean(vecs[:, :, None] * vecs[:, None, :], axis=0)
        ivars[m] = np.linalg.inv(vars[m])
        ivars[m] = 0.5 * (ivars[m] + ivars[m].T) # symmetrize

    # create mixture of M-choose-K times K! Gaussians (OMG!)
    Kfac = factorial(K)
    McKKfac = choose(M, K) * Kfac
    KD = K * D
    bigamps = np.zeros(McKKfac)
    bigmeans = np.zeros((McKKfac, KD))
    bigvars = np.zeros((McKKfac, KD, KD))
    bigivars = np.zeros((McKKfac, KD, KD))
    for i, p in enumerate(it.permutations(range(M), K)):
        bigamps[i] = 1.
        for k in range(K):
            bigamps[i] *= amps[p[k]]
            bigmeans[i, k * D : k * D + D] = means[p[k]]
            bigvars[i, k * D : k * D + D, k * D : k * D + D] = vars[p[k]]
            bigivars[i, k * D : k * D + D, k * D : k * D + D] = ivars[p[k]]
    bigamps /= McKKfac

    return mixture_of_gaussians(bigamps, bigmeans, bigvars, bigivars)

def get_log_prior(KD):
    """
    Return a "spherical" 1-Gaussian, `KD`-dimensional prior.
    """
    bigamps = np.ones((1,))
    bigmeans = np.ones((1, KD))
    foo = 1.
    bigvars = foo * np.eye(KD).reshape((1, KD, KD))
    bigivars = (1. / foo) * np.eye(KD).reshape((1, KD, KD))
    return mixture_of_gaussians(bigamps, bigmeans, bigvars, bigivars)

def hogg_savefig(fn):
    print("writing ", fn)
    return plt.savefig(fn)

if __name__ == "__main__":
    import pylab as plt
    import corner

    np.random.seed(42)
    tM, tK, tD = 7, 5, 6
    ln_prior = get_log_prior(tK * tD)
    ln_like = get_log_likelihood(tM, tK, tD)
    ln_post = ln_prior * ln_like # ARGH TERRIBLE TIMES
    print("FML:", np.exp(ln_post.log_fully_marginalized_value() - ln_prior.log_fully_marginalized_value()))
    xds = np.arange(-3., 3., 0.01)
    xs = np.zeros((len(xds), tK * tD))
    xs[:,0] = xds
    ln_ps1 = np.array([ln_prior(x) + ln_like(x) for x in xs])
    ln_ps2 = np.array([ln_post(x) for x in xs])
    plt.clf()
    plt.plot(xds, np.exp(ln_ps1 - np.max(ln_ps1)), "k-")
    plt.plot(xds, np.exp(ln_ps2 - np.max(ln_ps1)), "r--")
    hogg_savefig("cd.png")
    for d in range(ln_like.D):
        ln_Ls = np.array([ln_like(x, d=d) for x in xds])
        ln_ps = np.array([ln_post(x, d=d) for x in xds])
        plt.clf()
        plt.plot(xds, np.exp(ln_ps - np.max(ln_ps)), "k-")
        plt.plot(xds, np.exp(ln_Ls - np.max(ln_Ls)), "r--")
        hogg_savefig("cd{:04d}.png".format(d))
    samples = np.array([ln_post.draw_sample() for t in range(2 ** 15)])
    figure = corner.corner(samples)
    cfn = "corner.png"
    print("writing ", cfn)
    figure.savefig(cfn)
