"""
Posterior Predictive Check Plot
===============================

_thumb: .6, .5
"""
import arviz as az
import numpy as np
import pymc3 as pm

az.style.use('arviz-darkgrid')

# Data of the Eight Schools Model
J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])


with pm.Model() as centered_eight:
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sd=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
    centered_eight_trace = pm.sample()

with centered_eight:
    ppc_samples = pm.sample_ppc(centered_eight_trace, samples=100)

az.ppcplot(y, ppc_samples)
