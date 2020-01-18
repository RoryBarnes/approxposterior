#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Fit a parabola: y = ax^2 + b

@author: Rory Barnes [University of Washington, Seattle], 2020
@email: rory@astro.washington.edu

"""
import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
import corner
import george

from approxposterior import approx, gpUtils as gpu

aprior_min = -5
aprior_max = 5
bprior_min = -5
bprior_max = 5

# Define the loglikelihood function
def LogLikelihood(theta, x, obs, obserr):

    # Model parameters
    theta = np.array(theta)
    a, b = theta

    # Model predictions given parameters
    model = a*x*x + b

    # Likelihood of data given model parameters
    return -0.5*np.sum((obs-model)**2/obserr**2)

# Define the logprior function
def LogPrior(theta):

    # Model parameters
    theta = np.array(theta)
    a, b = theta

    # Probability of model parameters: flat prior
    if aprior_min < a < aprior_max and bprior_min < b < bprior_max:
        return 0.0
    return -np.inf

# Define logprobability function: l(D|theta) * p(theta)
# Note: use this for emcee, not approxposterior!
def LogProbability(theta, x, obs, obserr):

    lp = LogPrior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + LogLikelihood(theta, x, obs, obserr)

# Matrix of (a,b) for sampling the prior
def PriorSample(n):
    """
    docs

    Parameters
    ----------
    n : int
        Number of samples

    Returns
    -------
    sample : floats
        n x 3 array of floats samples from the prior
    """

    # Sample model parameters given prior distributions
    a = np.random.uniform(low=aprior_min, high=aprior_max, size=(n))
    b = np.random.uniform(low=bprior_min, high=bprior_max, size=(n))

    return np.array([a,b]).T

# Plot the fake data and the actual parabola
# Set seed for reproducibility.
seed = 42
np.random.seed(seed)

# Choose the "true" parameters.
aTrue = 3.77
bTrue = 2.77

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))-5
obserr = 5 # Amplitude of noise term
obs = aTrue*x*x + bTrue # True model
obs += obserr * np.random.randn(N) # Add some random noise

# Now plot it to see what the data looks like
fig, ax = plt.subplots(figsize=(8,8))

ax.errorbar(x, obs, yerr=obserr, fmt=".k", capsize=0)
x0 = np.linspace(-10, 10, 500)
ax.plot(x0, aTrue*x0*x0+bTrue, "k", alpha=0.3, lw=3)
ax.set_xlim(-5, 5)
ax.set_ylim(0,100)
ax.set_xlabel("x")
ax.set_ylabel("obs");

# Save figure
fig.savefig("ParabolaTrue.png", bbox_inches="tight")

# Now retrieve a and b with approxposterior

# Define algorithm parameters
iTrainInit = 20                           # Initial size of training set
iNewPoints = 10                            # Number of new points to find each iteration
iMaxIter = 5                          # Maximum number of iterations
# Prior bounds for a and b
bounds = [(aprior_min,aprior_max), (bprior_min,bprior_max)]
algorithm = "bape"                # Use the Kandasamy et al. (2017) formalism
seed = 57                         # RNG seed
np.random.seed(seed)

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc parameters

# Sample design points from prior
#theta = lh.rosenbrockSample(m0)
theta = np.array(PriorSample(iTrainInit))

# Evaluate forward model log likelihood + lnprior for each theta
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = LogLikelihood(theta[ii],x,obs,obserr) + LogPrior(theta[ii])

# Default GP with an ExpSquaredKernel
gp = gpu.defaultGP(theta, y, white_noise=-12)

# Initialize object using the Wang & Li (2018) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=LogPrior,
                            lnlike=LogLikelihood,
                            priorSample=PriorSample,
                            bounds=bounds,
                            algorithm=algorithm)

# Run!
ap.run(m=iNewPoints, nmax=iMaxIter, estBurnin=True, nGPRestarts=3, mcmcKwargs=mcmcKwargs,
       cache=False, samplerKwargs=samplerKwargs, verbose=True, thinChains=False,
       onlyLastMCMC=True)

# Check out the final posterior distribution!
import corner

# Load in chain from last iteration
samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

# Plot where forward model was evaluated - uncomment to plot!
fig.axes[2].scatter(ap.theta[iTrainInit:,0], ap.theta[iTrainInit:,1], s=10, color="red", zorder=20)

# Save figure
fig.savefig("finalPosterior.png", bbox_inches="tight")
