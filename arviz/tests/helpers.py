# pylint: disable=redefined-outer-name, comparison-with-callable
"""Test helper functions."""
import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version

from ..data import InferenceData, from_dict

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
    }
    log_likelihood = {
        "y": np.random.randn(nchains, ndraws, data["J"]),
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
        log_likelihood=log_likelihood,
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
    }
    log_likelihood = {
        "y": np.random.randn(nchains, ndraws, ndim1, ndim2),
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
        log_likelihood=log_likelihood,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data={"y": data["y"]},
        dims={"y": ["dim1", "dim2"], "log_likelihood": ["dim1", "dim2"]},
        coords={"dim1": range(ndim1), "dim2": range(ndim2)},
    )
    return model


def create_data_random(groups=None, seed=10):
    """Create InferenceData object using random data."""
    if groups is None:
        groups = ["posterior", "sample_stats", "observed_data", "posterior_predictive"]
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(4, 500, 8))
    idata_dict = dict(
        posterior={"a": data[..., 0], "b": data},
        sample_stats={"a": data[..., 0], "b": data},
        observed_data={"b": data[0, 0, :]},
        posterior_predictive={"a": data[..., 0], "b": data},
        prior={"a": data[..., 0], "b": data},
        prior_predictive={"a": data[..., 0], "b": data},
        warmup_posterior={"a": data[..., 0], "b": data},
        warmup_posterior_predictive={"a": data[..., 0], "b": data},
        warmup_prior={"a": data[..., 0], "b": data},
    )
    idata = from_dict(
        **{group: ary for group, ary in idata_dict.items() if group in groups}, save_warmup=True
    )
    return idata


@pytest.fixture()
def data_random():
    """Fixture containing InferenceData object using random data."""
    idata = create_data_random()
    return idata


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


def check_multiple_attrs(
    test_dict: Dict[str, List[str]], parent: InferenceData
) -> List[Union[str, Tuple[str, str]]]:
    """Perform multiple hasattr checks on InferenceData objects.

    It is thought to first check if the parent object contains a given dataset,
    and then (if present) check the attributes of the dataset.

    Given the output of the function, all mismatches between expectation and reality can
    be retrieved: a single string indicates a group mismatch and a tuple of strings
    ``(group, var)`` indicates a mismatch in the variable ``var`` of ``group``.

    Parameters
    ----------
    test_dict: dict of {str : list of str}
        Its structure should be `{dataset1_name: [var1, var2], dataset2_name: [var]}`.
        A ``~`` at the beginning of a dataset or variable name indicates the name NOT
        being present must be asserted.
    parent: InferenceData
        InferenceData object on which to check the attributes.

    Returns
    -------
    list
        List containing the failed checks. It will contain either the dataset_name or a
        tuple (dataset_name, var) for all non present attributes.

    Examples
    --------
    The output below indicates that ``posterior`` group was expected but not found, and
    variables ``a`` and ``b``:

        ["posterior", ("prior", "a"), ("prior", "b")]

    Another example could be the following:

        [("posterior", "a"), "~observed_data", ("sample_stats", "~log_likelihood")]

    In this case, the output indicates that variable ``a`` was not found in ``posterior``
    as it was expected, however, in the other two cases, the preceding ``~`` (kept from the
    input negation notation) indicates that ``observed_data`` group should not be present
    but was found in the InferenceData and that ``log_likelihood`` variable was found
    in ``sample_stats``, also against what was expected.

    """
    failed_attrs: List[Union[str, Tuple[str, str]]] = []
    for dataset_name, attributes in test_dict.items():
        if dataset_name.startswith("~"):
            if hasattr(parent, dataset_name[1:]):
                failed_attrs.append(dataset_name)
        elif hasattr(parent, dataset_name):
            dataset = getattr(parent, dataset_name)
            for attribute in attributes:
                if attribute.startswith("~"):
                    if hasattr(dataset, attribute[1:]):
                        failed_attrs.append((dataset_name, attribute))
                elif not hasattr(dataset, attribute):
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
    mcmc.sampler._postprocess_fn = None  # pylint: disable=protected-access
    mcmc.sampler._potential_fn = None  # pylint: disable=protected-access
    mcmc.sampler._potential_fn_gen = None  # pylint: disable=protected-access
    mcmc._cache = {}  # pylint: disable=protected-access
    return mcmc


def tfp_schools_model(num_schools, treatment_stddevs):
    """Non-centered eight schools model for tfp."""
    import tensorflow as tf
    import tensorflow_probability.python.edward2 as ed

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
    import tensorflow as tf
    import tensorflow_probability as tfp
    import tensorflow_probability.python.edward2 as ed

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
        import pystan  # pylint: disable=import-error

        stan_model = pystan.StanModel(model_code=schools_code)
        fit = stan_model.sampling(
            data=data,
            iter=draws + 500,
            warmup=500,
            chains=chains,
            check_hmc_diagnostics=False,
            control=dict(adapt_engaged=False),
        )
    else:
        import stan  # pylint: disable=import-error

        stan_model = stan.build(schools_code, data=data)
        fit = stan_model.sample(
            num_chains=chains, num_samples=draws, num_warmup=500, save_warmup=False
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
                try:
                    _log.info("Generating and caching %s", fname)
                    cloudpickle.dump(func(eight_schools_data, draws, chains), buff)
                except AttributeError as err:
                    raise AttributeError(f"Failed caching {library_name}") from err

        with gzip.open(path, "rb") as buff:
            _log.info("Loading %s from cache", fname)
            models[library.__name__] = cloudpickle.load(buff)

    return models


def pystan_version():
    """Check PyStan version.

    Returns
    -------
    int
        Major version number

    """
    try:
        import pystan  # pylint: disable=import-error

        version = int(pystan.__version__[0])
    except ImportError:
        try:
            import stan  # pylint: disable=import-error

            version = int(stan.__version__[0])
        except ImportError:
            version = None
    return version


def test_precompile_models(eight_schools_params, draws, chains):
    """Precompile model files."""
    load_cached_models(eight_schools_params, draws, chains)


def running_on_ci() -> bool:
    """Return True if running on CI machine."""
    return os.environ.get("ARVIZ_CI_MACHINE") is not None


def importorskip(
    modname: str, minversion: Optional[str] = None, reason: Optional[str] = None
) -> Any:
    """Import and return the requested module ``modname``.

        Doesn't allow skips on CI machine.
        Borrowed and modified from ``pytest.importorskip``.
    :param str modname: the name of the module to import
    :param str minversion: if given, the imported module's ``__version__``
        attribute must be at least this minimal version, otherwise the test is
        still skipped.
    :param str reason: if given, this reason is shown as the message when the
        module cannot be imported.
    :returns: The imported module. This should be assigned to its canonical
        name.
    Example::
        docutils = pytest.importorskip("docutils")
    """
    # ARVIZ_CI_MACHINE is True if tests run on CI, where ARVIZ_CI_MACHINE env variable exists
    ARVIZ_CI_MACHINE = running_on_ci()
    if ARVIZ_CI_MACHINE:
        import warnings

        compile(modname, "", "eval")  # to catch syntaxerrors

        with warnings.catch_warnings():
            # make sure to ignore ImportWarnings that might happen because
            # of existing directories with the same name we're trying to
            # import but without a __init__.py file
            warnings.simplefilter("ignore")
            __import__(modname)
        mod = sys.modules[modname]
        if minversion is None:
            return mod
        verattr = getattr(mod, "__version__", None)
        if minversion is not None:
            if verattr is None or Version(verattr) < Version(minversion):
                raise Skipped(
                    "module %r has __version__ %r, required is: %r"
                    % (modname, verattr, minversion),
                    allow_module_level=True,
                )
        return mod
    else:
        return pytest.importorskip(modname=modname, minversion=minversion, reason=reason)
