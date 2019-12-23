# pylint: disable=redefined-outer-name, comparison-with-callable
"""Test helper functions."""
import gzip
import importlib
import os
import pickle
import sys
import logging
import pytest
import numpy as np

from ..data import from_dict


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


def create_model(seed=10):
    """Create model with fake data."""
    np.random.seed(seed)
    nchains = 4
    ndraws = 500
    data = {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }
    posterior = {
        "mu": np.random.randn(nchains, ndraws),
        "tau": abs(np.random.randn(nchains, ndraws)),
        "eta": np.random.randn(nchains, ndraws, data["J"]),
        "theta": np.random.randn(nchains, ndraws, data["J"]),
    }
    posterior_predictive = {"y": np.random.randn(nchains, ndraws, len(data["y"]))}
    sample_stats = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": np.random.randn(nchains, ndraws) > 0.90,
        "max_depth": np.random.randn(nchains, ndraws) > 0.90,
        "log_likelihood": np.random.randn(nchains, ndraws, data["J"]),
    }
    prior = {
        "mu": np.random.randn(nchains, ndraws) / 2,
        "tau": abs(np.random.randn(nchains, ndraws)) / 2,
        "eta": np.random.randn(nchains, ndraws, data["J"]) / 2,
        "theta": np.random.randn(nchains, ndraws, data["J"]) / 2,
    }
    prior_predictive = {"y": np.random.randn(nchains, ndraws, len(data["y"])) / 2}
    sample_stats_prior = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": (np.random.randn(nchains, ndraws) > 0.95).astype(int),
    }
    model = from_dict(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        sample_stats=sample_stats,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data={"y": data["y"]},
        dims={"y": ["obs_dim"], "log_likelihood": ["obs_dim"]},
        coords={"obs_dim": range(data["J"])},
    )
    return model


def create_multidimensional_model(seed=10):
    """Create model with fake data."""
    np.random.seed(seed)
    nchains = 4
    ndraws = 500
    ndim1 = 5
    ndim2 = 7
    data = {
        "y": np.random.normal(size=(ndim1, ndim2)),
        "sigma": np.random.normal(size=(ndim1, ndim2)),
    }
    posterior = {
        "mu": np.random.randn(nchains, ndraws),
        "tau": abs(np.random.randn(nchains, ndraws)),
        "eta": np.random.randn(nchains, ndraws, ndim1, ndim2),
        "theta": np.random.randn(nchains, ndraws, ndim1, ndim2),
    }
    posterior_predictive = {"y": np.random.randn(nchains, ndraws, ndim1, ndim2)}
    sample_stats = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": np.random.randn(nchains, ndraws) > 0.90,
        "log_likelihood": np.random.randn(nchains, ndraws, ndim1, ndim2),
    }
    prior = {
        "mu": np.random.randn(nchains, ndraws) / 2,
        "tau": abs(np.random.randn(nchains, ndraws)) / 2,
        "eta": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2,
        "theta": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2,
    }
    prior_predictive = {"y": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2}
    sample_stats_prior = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": (np.random.randn(nchains, ndraws) > 0.95).astype(int),
    }
    model = from_dict(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        sample_stats=sample_stats,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data={"y": data["y"]},
        dims={"y": ["dim1", "dim2"], "log_likelihood": ["dim1", "dim2"]},
        coords={"dim1": range(ndim1), "dim2": range(ndim2)},
    )
    return model


@pytest.fixture(scope="module")
def models():
    """Fixture containing 2 mock inference data instances for testing."""
    # blank line to keep black and pydocstyle happy

    class Models:
        model_1 = create_model(seed=10)
        model_2 = create_model(seed=11)

    return Models()


@pytest.fixture(scope="module")
def multidim_models():
    """Fixture containing 2 mock inference data instances with multidimensional data for testing."""
    # blank line to keep black and pydocstyle happy

    class Models:
        model_1 = create_multidimensional_model(seed=10)
        model_2 = create_multidimensional_model(seed=11)

    return Models()


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
    prior_mu = -((mu / 10) ** 2)  # normal prior, loc=0, scale=10
    prior_eta = -np.sum(eta ** 2)  # normal prior, loc=0, scale=1
    return prior_mu + prior_tau + prior_eta


def _emcee_lnprob(theta, y, sigma):
    """Proper function to allow pickling."""
    mu, tau, eta = theta[0], theta[1], theta[2:]
    prior = _emcee_lnprior(theta)
    like_vect = -(((mu + tau * eta - y) / sigma) ** 2)
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

    pos = np.random.normal(size=(chains, ndim))
    pos[:, 1] = np.absolute(pos[:, 1])  #  pylint: disable=unsupported-assignment-operation

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


# pylint:disable=no-member,no-value-for-parameter,invalid-name
def _pyro_noncentered_model(J, sigma, y=None):
    import pyro
    import pyro.distributions as dist

    mu = pyro.sample("mu", dist.Normal(0, 5))
    tau = pyro.sample("tau", dist.HalfCauchy(5))
    with pyro.plate("J", J):
        eta = pyro.sample("eta", dist.Normal(0, 1))
        theta = mu + tau * eta
        return pyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def pyro_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation in Pyro."""
    import torch
    from pyro.infer import MCMC, NUTS

    y = torch.from_numpy(data["y"]).float()
    sigma = torch.from_numpy(data["sigma"]).float()

    nuts_kernel = NUTS(_pyro_noncentered_model, jit_compile=True, ignore_jit_warnings=True)
    posterior = MCMC(nuts_kernel, num_samples=draws, warmup_steps=draws, num_chains=chains)
    posterior.run(data["J"], sigma, y)

    # This block lets the posterior be pickled
    posterior.sampler = None
    posterior.kernel.potential_fn = None
    return posterior


# pylint:disable=no-member,no-value-for-parameter,invalid-name
def _numpyro_noncentered_model(J, sigma, y=None):
    import numpyro
    import numpyro.distributions as dist

    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        eta = numpyro.sample("eta", dist.Normal(0, 1))
        theta = mu + tau * eta
        return numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def numpyro_schools_model(data, draws, chains):
    """Centered eight schools implementation in NumPyro."""
    from jax.random import PRNGKey
    from numpyro.infer import MCMC, NUTS

    mcmc = MCMC(
        NUTS(_numpyro_noncentered_model),
        num_warmup=draws,
        num_samples=draws,
        num_chains=chains,
        chain_method="sequential",
    )
    mcmc.run(PRNGKey(0), extra_fields=("num_steps", "energy"), **data)

    # This block lets the posterior be pickled
    mcmc.sampler._sample_fn = None  # pylint: disable=protected-access
    mcmc.sampler._init_fn = None  # pylint: disable=protected-access
    mcmc.sampler._constrain_fn = None  # pylint: disable=protected-access
    mcmc._cache = {}  # pylint: disable=protected-access
    return mcmc


def tfp_schools_model(num_schools, treatment_stddevs):
    """Non-centered eight schools model for tfp."""
    import tensorflow_probability.python.edward2 as ed
    import tensorflow as tf

    if int(tf.__version__[0]) > 1:
        import tensorflow.compat.v1 as tf  # pylint: disable=import-error

        tf.disable_v2_behavior()

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

    if int(tf.__version__[0]) > 1:
        import tensorflow.compat.v1 as tf  # pylint: disable=import-error

        tf.disable_v2_behavior()

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
            data=data,
            iter=draws,
            warmup=0,
            chains=chains,
            check_hmc_diagnostics=False,
            control=dict(adapt_engaged=False),
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
        ("pyro", pyro_noncentered_schools),
        ("numpyro", numpyro_schools_model),
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
