import pytest
import emcee
from arviz import from_emcee
import numpy as np
import os


def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2*np.exp(2*log_f)
    return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))


def log_prior(theta):
    m, b, log_f = theta
    if (-5.0 < m < 0.5) and (0.0 < b < 10.0) and (-10.0 < log_f < 1.0):
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


def test_emcee_reader_data():
    chains = 40
    draws = 1500
    # Choose the "true" parameters.
    m_true = -0.9594
    b_true = 4.294
    f_true = 0.534

    # make reproducible
    np.random.seed(0)

    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(10*np.random.rand(N))
    yerr = 0.1+0.5*np.random.rand(N)
    y = m_true*x+b_true
    y += np.abs(f_true*y) * np.random.randn(N)
    y += yerr * np.random.randn(N)

    # Run the sampler saving the results in HDFBackend
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, "saved_models")
    filepath = os.path.join(data_directory, "reader_testfile.h5")
    pos = np.array([-1,4,.5]) + 1e-4*np.random.randn(chains, 3)
    nwalkers, ndim = pos.shape
    backend = emcee.backends.HDFBackend(filepath)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr), backend=backend)
    sampler.run_mcmc(pos, draws);

    # read data form HDFBackend to arviz
    del sampler
    reader = emcee.backends.HDFBackend(filepath)
    inference_data = from_emcee(reader, var_names=["b", "m", "ln(f)"])
    assert hasattr(inference_data, "posterior")
    assert hasattr(inference_data.posterior, "ln(f)")
    assert hasattr(inference_data.posterior, "b")
    assert hasattr(inference_data.posterior, "m")
