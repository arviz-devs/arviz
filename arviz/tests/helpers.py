"""Test helper functions."""
import os
import pickle
import sys

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
import pystan
import scipy.optimize as op
import torch


class BaseArvizTest():
    """Base class for running arviz tests."""

    @classmethod
    def setup_class(cls):
        """Run once for the class.

        This has to exist so subclasses can inherit.
        """
        pass

    @classmethod
    def teardown_class(cls):
        """Teardown at end of tests.

        This has to exist so subclasses can inherit.
        """
        pass

    def setup_method(self):
        """Run for every test."""
        np.random.seed(1)

    def teardown_method(self):
        """Run for every test."""
        plt.close('all')


def _emcee_neg_lnlike(theta, x, y, yerr):
    """Proper function to allow pickling."""
    slope, intercept, lnf = theta
    model = slope * x + intercept
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return 0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def _emcee_lnprior(theta):
    """Proper function to allow pickling."""
    slope, intercept, lnf = theta
    if -5.0 < slope < 0.5 and 0.0 < intercept < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


def _emcee_lnprob(theta, x, y, yerr):
    """Proper function to allow pickling."""
    logp = _emcee_lnprior(theta)
    if not np.isfinite(logp):
        return -np.inf
    return logp - _emcee_neg_lnlike(theta, x, y, yerr)


def emcee_linear_model(data, draws, chains):
    """Linear model fit in emcee.

    Note that the data is unused, but included to fit the pattern
    from other libraries.

    From http://dfm.io/emcee/current/user/line/
    """
    del data
    chains = 10 * chains  # emcee is sad with too few walkers

    # Choose the "true" parameters.
    m_true = -0.9594
    b_true = 4.294
    f_true = 0.534

    # make reproducible
    np.random.seed(0)

    # Generate some synthetic data from the model.
    num_data = 50
    x = np.sort(10*np.random.rand(num_data))
    yerr = 0.1+0.5*np.random.rand(num_data)
    y = m_true*x+b_true
    y += np.abs(f_true*y) * np.random.randn(num_data)
    y += yerr * np.random.randn(num_data)

    result = op.minimize(_emcee_neg_lnlike, [m_true, b_true, np.log(f_true)],
                         args=(x, y, yerr))

    ndim = result['x'].shape[0]
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for _ in range(chains)]

    sampler = emcee.EnsembleSampler(chains, ndim, _emcee_lnprob, args=(x, y, yerr))

    sampler.run_mcmc(pos, draws)
    return sampler


def eight_schools_params():
    """Share setup for eight schools."""
    return {
        'J': 8,
        'y': np.array([28., 8., -3., 7., -1., 1., 18., 12.]),
        'sigma': np.array([15., 10., 16., 11., 9., 11., 10., 18.]),
    }


# pylint:disable=no-member,no-value-for-parameter
def _pyro_centered_model(sigma):
    """Centered model setup."""
    mu = pyro.sample('mu', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    tau = pyro.sample('tau', dist.HalfCauchy(scale=25 * torch.ones(1)))

    theta = pyro.sample('theta',
                        dist.Normal(
                            mu * torch.ones(8),
                            tau * torch.ones(8)))

    return pyro.sample("obs", dist.Normal(theta, sigma))


def _pyro_conditioned_model(model, sigma, y):
    """Condition the model."""
    return pyro.poutine.condition(model, data={"obs": y})(sigma)


def pyro_centered_schools(data, draws, chains):
    """Centered eight schools implementation in Pyro.

    Note there is not really a deterministic node in pyro, so I do not
    know how to do a non-centered implementation.
    """
    del chains
    y = torch.Tensor(data['y']).type(torch.Tensor)
    sigma = torch.Tensor(data['sigma']).type(torch.Tensor)

    nuts_kernel = NUTS(_pyro_conditioned_model, adapt_step_size=True)
    posterior = MCMC(  # pylint:disable=not-callable
        nuts_kernel,
        num_samples=draws,
        warmup_steps=500,
    ).run(_pyro_centered_model, sigma, y)

    # This block lets the posterior be pickled
    for trace in posterior.exec_traces:
        for node in trace.nodes.values():
            node.pop('fn', None)
    posterior.kernel = None
    posterior.run = None
    posterior.logger = None
    return posterior


def pystan_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation for pystan."""
    schools_code = """
        data {
            int<lower=0> J;
            real y[J];
            real<lower=0> sigma[J];
        }

        parameters {
            real mu;
            real<lower=0> tau;
            real theta_tilde[J];
        }

        transformed parameters {
            real theta[J];
            for (j in 1:J)
                theta[j] = mu + tau * theta_tilde[j];
        }

        model {
            mu ~ normal(0, 5);
            tau ~ cauchy(0, 5);
            theta_tilde ~ normal(0, 1);
            y ~ normal(theta, sigma);
        }

        generated quantities {
            vector[J] log_lik;
            vector[J] y_hat;
            for (j in 1:J) {
                log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
                y_hat[j] = normal_rng(theta[j], sigma[j]);
            }
        }
    """
    stan_model = pystan.StanModel(model_code=schools_code)
    fit = stan_model.sampling(data=data,
                              iter=draws,
                              warmup=0,
                              chains=chains)
    return stan_model, fit


def pymc3_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation for pymc3."""
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sd=5)
        tau = pm.HalfCauchy('tau', beta=5)
        theta_tilde = pm.Normal('theta_tilde', mu=0, sd=1, shape=data['J'])
        theta = pm.Deterministic('theta', mu + tau * theta_tilde)
        pm.Normal('obs', mu=theta, sd=data['sigma'], observed=data['y'])
        trace = pm.sample(draws, chains=chains)
    return model, trace


def load_cached_models(draws, chains):
    """Load pymc3, pystan, and emcee models from pickle."""
    here = os.path.dirname(os.path.abspath(__file__))
    data = eight_schools_params()
    supported = (
        (pystan, pystan_noncentered_schools),
        (pm, pymc3_noncentered_schools),
        (emcee, emcee_linear_model),
        (pyro, pyro_centered_schools),
    )
    data_directory = os.path.join(here, 'saved_models')
    models = {}
    for library, func in supported:
        py_version = sys.version_info
        fname = '{0.major}.{0.minor}_{1.__name__}_{1.__version__}_{2}_{3}.pkl'.format(
            py_version, library, draws, chains

        )
        path = os.path.join(data_directory, fname)
        if not os.path.exists(path):
            with open(path, 'wb') as buff:
                pickle.dump(func(data, draws, chains), buff)
        with open(path, 'rb') as buff:
            models[library.__name__] = pickle.load(buff)
    return models
