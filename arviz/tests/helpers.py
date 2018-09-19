"""Test helper functions."""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pystan


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


def eight_schools_params():
    """Share setup for eight schools."""
    return {
        'J': 8,
        'y': np.array([28., 8., -3., 7., -1., 1., 18., 12.]),
        'sigma': np.array([15., 10., 16., 11., 9., 11., 10., 18.]),
    }


def pystan_noncentered_schools(data, draws, chains):
    """Non-centered eight schools implementation for pystan."""
    schools_code = '''
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
    '''
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
    """Load pymc3 and pystan models from pickle."""
    here = os.path.dirname(os.path.abspath(__file__))
    data = eight_schools_params()
    supported = (
        (pystan, pystan_noncentered_schools),
        (pm, pymc3_noncentered_schools),
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
