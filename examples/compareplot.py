"""
Compare Plot
============

_thumb: .5, .5
"""
import arviz as az
import numpy as np
import pymc3 as pm

az.style.use('arviz-darkgrid')

# Data of the Eight Schools Model
J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])


with pm.Model('Centered Eight Schools') as centered_eight:
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sd=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
    centered_eight_trace = pm.sample()


with pm.Model('Non-Centered Eight Schools') as non_centered:
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta_tilde = pm.Normal('theta_t', mu=0, sd=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_tilde)
    obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
    non_centered_eight_trace = pm.sample()


model_compare = az.compare({
    centered_eight: centered_eight_trace,
    non_centered: non_centered_eight_trace
})

az.compareplot(model_compare, figsize=(12, 4))