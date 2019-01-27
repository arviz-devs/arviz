"""Test helper functions."""
from collections import OrderedDict
import os
import pickle
import sys
import logging
import pytest

import emcee
import numpy as np
import pymc3 as pm
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS

try:
    import pystan
except ImportError:
    import stan as pystan

import tensorflow_probability as tfp
import tensorflow_probability.python.edward2 as ed
import scipy.optimize as op
import torch
import tensorflow as tf


_log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def eight_schools_params():
    """Share setup for eight schools."""
    return {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }


def emcee_version():
    """Check emcee version.

    Returns
    -------
    int
        Major version number
    """
    return int(emcee.__version__[0])


# pylint: disable=invalid-name
needs_emcee3 = pytest.mark.skipif(emcee_version() < 3, reason="emcee3 required")


def _emcee_neg_lnlike(theta, x, y, yerr):
    """Proper function to allow pickling."""
    slope, intercept, lnf = theta
    model = slope * x + intercept
    inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
    return 0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


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
    x = np.sort(10 * np.random.rand(num_data))
    yerr = 0.1 + 0.5 * np.random.rand(num_data)
    y = m_true * x + b_true
    y += np.abs(f_true * y) * np.random.randn(num_data)
    y += yerr * np.random.randn(num_data)

    result = op.minimize(_emcee_neg_lnlike, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))

    ndim = result["x"].shape[0]
    pos = [result["x"] + 1e-4 * np.random.randn(ndim) for _ in range(chains)]

    if emcee_version() < 3:
        sampler = emcee.EnsembleSampler(chains, ndim, _emcee_lnprob, args=(x, y, yerr))
        # pylint: enable=unexpected-keyword-arg
        sampler.run_mcmc(pos, draws)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "saved_models")
        filepath = os.path.join(data_directory, "reader_testfile.h5")
        backend = emcee.backends.HDFBackend(filepath)  # pylint: disable=no-member
        backend.reset(chains, ndim)
        # pylint: disable=unexpected-keyword-arg
        sampler = emcee.EnsembleSampler(
            chains, ndim, _emcee_lnprob, args=(x, y, yerr), backend=backend
        )
        # pylint: enable=unexpected-keyword-arg
        sampler.run_mcmc(pos, draws, store=True)
    return sampler


# pylint:disable=no-member,no-value-for-parameter
def _pyro_centered_model(sigma):
    """Centered model setup."""
    mu = pyro.sample("mu", dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    tau = pyro.sample("tau", dist.HalfCauchy(scale=25 * torch.ones(1)))

    theta = pyro.sample("theta", dist.Normal(mu * torch.ones(8), tau * torch.ones(8)))

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
    y = torch.Tensor(data["y"]).type(torch.Tensor)
    sigma = torch.Tensor(data["sigma"]).type(torch.Tensor)

    nuts_kernel = NUTS(_pyro_conditioned_model, adapt_step_size=True)
    posterior = MCMC(  # pylint:disable=not-callable
        nuts_kernel, num_samples=draws, warmup_steps=500
    ).run(_pyro_centered_model, sigma, y)

    # This block lets the posterior be pickled
    for trace in posterior.exec_traces:
        for node in trace.nodes.values():
            node.pop("fn", None)
    posterior.kernel = None
    posterior.run = None
    posterior.logger = None
    if hasattr(posterior, "sampler"):
        posterior.sampler = None
    return posterior


def tfp_schools_model(num_schools, treatment_stddevs):
    """Non-centered eight schools model for tfp."""
    avg_effect = ed.Normal(loc=0.0, scale=10.0, name="avg_effect")  # `mu`
    avg_stddev = ed.Normal(loc=5.0, scale=1.0, name="avg_stddev")  # `log(tau)`
    school_effects_standard = ed.Normal(
        loc=tf.zeros(num_schools), scale=tf.ones(num_schools), name="school_effects_standard"
    )  # `eta`
    school_effects = avg_effect + tf.exp(avg_stddev) * school_effects_standard  # `theta`
    treatment_effects = ed.Normal(
        loc=school_effects, scale=treatment_stddevs, name="treatment_effects"
    )  # `y`
    return treatment_effects


def tfp_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation for tfp."""
    del chains

    log_joint = ed.make_log_joint_fn(tfp_schools_model)

    def target_log_prob_fn(avg_effect, avg_stddev, school_effects_standard):
        """Unnormalized target density as a function of states."""
        return log_joint(
            num_schools=data["J"],
            treatment_stddevs=data["sigma"].astype(np.float32),
            avg_effect=avg_effect,
            avg_stddev=avg_stddev,
            school_effects_standard=school_effects_standard,
            treatment_effects=data["y"].astype(np.float32),
        )

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=draws,
        num_burnin_steps=500,
        current_state=[
            tf.zeros([], name="init_avg_effect"),
            tf.zeros([], name="init_avg_stddev"),
            tf.ones([data["J"]], name="init_school_effects_standard"),
        ],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn, step_size=0.4, num_leapfrog_steps=3
        ),
    )

    with tf.Session() as sess:
        [states_, _] = sess.run([states, kernel_results])

    return tfp_schools_model, states_


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
            real eta[J];
        }

        transformed parameters {
            real theta[J];
            for (j in 1:J)
                theta[j] = mu + tau * eta[j];
        }

        model {
            mu ~ normal(0, 5);
            tau ~ cauchy(0, 5);
            eta ~ normal(0, 1);
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
    if pystan_version() == 2:
        stan_model = pystan.StanModel(model_code=schools_code)
        fit = stan_model.sampling(
            data=data, iter=draws, warmup=0, chains=chains, check_hmc_diagnostics=False
        )
    else:
        # hard code schools data
        # bug in PyStan3 preview. It modify data in-place
        data = {
            "J": 8,
            "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
            "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
        }
        stan_model = pystan.build(schools_code, data=data)
        fit = stan_model.sample(
            num_chains=chains, num_samples=draws, num_warmup=0, save_warmup=False
        )
    return stan_model, fit


def pymc3_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation for pymc3."""
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sd=5)
        tau = pm.HalfCauchy("tau", beta=5)
        eta = pm.Normal("eta", mu=0, sd=1, shape=data["J"])
        theta = pm.Deterministic("theta", mu + tau * eta)
        pm.Normal("obs", mu=theta, sd=data["sigma"], observed=data["y"])
        trace = pm.sample(draws, chains=chains)
    return model, trace


def load_cached_models(eight_schools_data, draws, chains):
    """Load pymc3, pystan, emcee, and pyro models from pickle."""
    here = os.path.dirname(os.path.abspath(__file__))
    supported = (
        (tfp, tfp_noncentered_schools),
        (pystan, pystan_noncentered_schools),
        (pm, pymc3_noncentered_schools),
        (emcee, emcee_linear_model),
        (pyro, pyro_centered_schools),
    )
    data_directory = os.path.join(here, "saved_models")
    models = {}

    for library, func in supported:
        if library.__name__ == "stan":
            # PyStan3 does not support pickling
            # httpstan caches models automatically
            _log.info("Generating and loading stan model")
            models["pystan"] = func(eight_schools_data, draws, chains)
            continue

        py_version = sys.version_info
        fname = "{0.major}.{0.minor}_{1.__name__}_{1.__version__}_{2}_{3}_{4}.pkl".format(
            py_version, library, sys.platform, draws, chains
        )

        path = os.path.join(data_directory, fname)
        if not os.path.exists(path):
            with open(path, "wb") as buff:
                _log.info("Generating and caching %s", fname)
                pickle.dump(func(eight_schools_data, draws, chains), buff)

        with open(path, "rb") as buff:
            _log.info("Loading %s from cache", fname)
            models[library.__name__] = pickle.load(buff)

    return models


def pystan_extract_unpermuted(fit, var_names=None):
    """Extract PyStan samples unpermuted.

    For PyStan 2.18+.
    Function returns everything as a float.
    """
    if var_names is None:
        var_names = fit.model_pars
    elif isinstance(var_names, str):
        var_names = [var_names]
    extract = fit.extract(var_names, permuted=False)
    return extract


def stan_extract_dict(fit, var_names=None):
    """Extract draws from PyStan3 fit.

    For PyStan 3.0a+
    Function returns everything as a float.
    """
    if var_names is None:
        var_names = fit.param_names
    elif isinstance(var_names, str):
        var_names = [var_names]
    var_names = list(var_names)

    data = OrderedDict()

    for var in var_names:
        if var in data:
            continue

        # in future fix the correct number of draws if fit.save_warmup is True
        new_shape = (*fit.dims[fit.param_names.index(var)], -1, fit.num_chains)
        values = fit._draws[fit._parameter_indexes(var), :]  # pylint: disable=protected-access
        values = values.reshape(new_shape, order="F")
        values = np.moveaxis(values, [-2, -1], [1, 0])
        data[var] = values

    return data


def pystan_version():
    """Check PyStan version.

    Returns
    -------
    int
        Major version number
    """
    return int(pystan.__version__[0])
