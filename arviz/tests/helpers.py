# pylint: disable=redefined-outer-name
"""Test helper functions."""
import gzip
import importlib
import os
import pickle
import sys
import logging
import pytest
import numpy as np

_log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def eight_schools_params():
    """Share setup for eight schools."""
    return {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }


@pytest.fixture(scope="module")
def draws():
    """Share default draw count."""
    return 500


@pytest.fixture(scope="module")
def chains():
    """Share default chain count."""
    return 2


def check_multiple_attrs(test_dict, parent):
    """Perform multiple hasattr checks on InferenceData objects.

    It is thought to first check if the parent object contains a given dataset,
    and then (if present) check the attributes of the dataset.

    Args
    ----
    test_dict: dict
        Its structure should be `{dataset1_name: [var1, var2], dataset2_name: [var]}`
    parent: InferenceData
        InferenceData object on which to check the attributes.

    Returns
    -------
    list
        List containing the failed checks. It will contain either the dataset_name or a
        tuple (dataset_name, var) for all non present attributes.

    """
    failed_attrs = []
    for dataset_name, attributes in test_dict.items():
        if hasattr(parent, dataset_name):
            dataset = getattr(parent, dataset_name)
            for attribute in attributes:
                if not hasattr(dataset, attribute):
                    failed_attrs.append((dataset_name, attribute))
        else:
            failed_attrs.append(dataset_name)
    return failed_attrs


def emcee_version():
    """Check emcee version.

    Returns
    -------
    int
        Major version number

    """
    import emcee

    return int(emcee.__version__[0])


def needs_emcee3_func():
    """Check if emcee3 is required."""
    # pylint: disable=invalid-name
    needs_emcee3 = pytest.mark.skipif(emcee_version() < 3, reason="emcee3 required")
    return needs_emcee3


def _emcee_lnprior(theta):
    """Proper function to allow pickling."""
    mu, tau, eta = theta[0], theta[1], theta[2:]
    # Half-cauchy prior, hwhm=25
    if tau < 0:
        return -np.inf
    prior_tau = -np.log(tau ** 2 + 25 ** 2)
    prior_mu = -(mu / 10) ** 2  # normal prior, loc=0, scale=10
    prior_eta = -np.sum(eta ** 2)  # normal prior, loc=0, scale=1
    return prior_mu + prior_tau + prior_eta


def _emcee_lnprob(theta, y, sigma):
    """Proper function to allow pickling."""
    mu, tau, eta = theta[0], theta[1], theta[2:]
    prior = _emcee_lnprior(theta)
    like_vect = -((mu + tau * eta - y) / sigma) ** 2
    like = np.sum(like_vect)
    return like + prior, (like_vect, np.random.normal((mu + tau * eta), sigma))


def emcee_schools_model(data, draws, chains):
    """Schools model in emcee."""
    import emcee

    chains = 10 * chains  # emcee is sad with too few walkers
    y = data["y"]
    sigma = data["sigma"]
    J = data["J"]  # pylint: disable=invalid-name
    ndim = J + 2

    # make reproducible
    np.random.seed(0)

    pos = np.random.normal(size=(chains, ndim))
    pos[:, 1] = np.absolute(pos[:, 1])

    if emcee_version() < 3:
        sampler = emcee.EnsembleSampler(chains, ndim, _emcee_lnprob, args=(y, sigma))
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
            chains, ndim, _emcee_lnprob, args=(y, sigma), backend=backend
        )
        # pylint: enable=unexpected-keyword-arg
        sampler.run_mcmc(pos, draws, store=True)
    return sampler


# pylint:disable=no-member,no-value-for-parameter
def _pyro_centered_model(sigma):
    """Centered model setup."""
    import pyro
    import torch
    import pyro.distributions as dist

    mu = pyro.sample("mu", dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    tau = pyro.sample("tau", dist.HalfCauchy(scale=25 * torch.ones(1)))

    theta = pyro.sample("theta", dist.Normal(mu * torch.ones(8), tau * torch.ones(8)))

    return pyro.sample("obs", dist.Normal(theta, sigma))


def _pyro_conditioned_model(model, sigma, y):
    """Condition the model."""
    import pyro

    return pyro.poutine.condition(model, data={"obs": y})(sigma)


def pyro_centered_schools(data, draws, chains):
    """Centered eight schools implementation in Pyro.

    Note there is not really a deterministic node in pyro, so I do not
    know how to do a non-centered implementation.
    """
    import torch
    from pyro.infer.mcmc import MCMC, NUTS

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
    import tensorflow_probability.python.edward2 as ed
    import tensorflow as tf

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
    import tensorflow_probability as tfp
    import tensorflow_probability.python.edward2 as ed
    import tensorflow as tf

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
        import pystan

        stan_model = pystan.StanModel(model_code=schools_code)
        fit = stan_model.sampling(
            data=data, iter=draws, warmup=0, chains=chains, check_hmc_diagnostics=False
        )
    else:
        import stan  # pylint: disable=import-error

        stan_model = stan.build(schools_code, data=data)
        fit = stan_model.sample(
            num_chains=chains, num_samples=draws, num_warmup=0, save_warmup=False
        )
    return stan_model, fit


def pymc3_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation for pymc3."""
    import pymc3 as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sd=5)
        tau = pm.HalfCauchy("tau", beta=5)
        eta = pm.Normal("eta", mu=0, sd=1, shape=data["J"])
        theta = pm.Deterministic("theta", mu + tau * eta)
        pm.Normal("obs", mu=theta, sd=data["sigma"], observed=data["y"])
        trace = pm.sample(draws, chains=chains)
    return model, trace


def library_handle(library):
    """Import a library and return the handle."""
    if library == "pystan":
        try:
            module = importlib.import_module("pystan")
        except ImportError:
            module = importlib.import_module("stan")
    else:
        module = importlib.import_module(library)
    return module


def load_cached_models(eight_schools_data, draws, chains, libs=None):
    """Load pymc3, pystan, emcee, and pyro models from pickle."""
    here = os.path.dirname(os.path.abspath(__file__))
    supported = (
        ("tensorflow_probability", tfp_noncentered_schools),
        ("pystan", pystan_noncentered_schools),
        ("pymc3", pymc3_noncentered_schools),
        ("emcee", emcee_schools_model),
        ("pyro", pyro_centered_schools),
    )
    data_directory = os.path.join(here, "saved_models")
    models = {}

    if isinstance(libs, str):
        libs = [libs]

    for library_name, func in supported:
        if libs is not None and library_name not in libs:
            continue
        library = library_handle(library_name)
        if library.__name__ == "stan":
            # PyStan3 does not support pickling
            # httpstan caches models automatically
            _log.info("Generating and loading stan model")
            models["pystan"] = func(eight_schools_data, draws, chains)
            continue

        py_version = sys.version_info
        fname = "{0.major}.{0.minor}_{1.__name__}_{1.__version__}_{2}_{3}_{4}.pkl.gzip".format(
            py_version, library, sys.platform, draws, chains
        )

        path = os.path.join(data_directory, fname)
        if not os.path.exists(path):
            with gzip.open(path, "wb") as buff:
                _log.info("Generating and caching %s", fname)
                pickle.dump(func(eight_schools_data, draws, chains), buff)

        with gzip.open(path, "rb") as buff:
            _log.info("Loading %s from cache", fname)
            models[library.__name__] = pickle.load(buff)

    return models


def pystan_version():
    """Check PyStan version.

    Returns
    -------
    int
        Major version number

    """
    try:
        import pystan
    except ImportError:
        import stan as pystan  # pylint: disable=import-error
    return int(pystan.__version__[0])
