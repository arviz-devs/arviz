# pylint: disable=no-member
import numpy as np
#import pytest

import pymc3 as pm
import pystan

from ..utils.xarray_utils import convert_to_xarray, PyMC3ToXarray, PyStanToXarray

def eight_schools_params():
    """Share setup for eight schools"""

    data = {
        'J': 8,
        'y': np.array([28., 8., -3., 7., -1., 1., 18., 12.]),
        'sigma': np.array([15., 10., 16., 11., 9., 11., 10., 18.]),
    }
    draws = 500
    chains = 2
    return data, draws, chains


def pystan_noncentered_schools(data, draws, chains):
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
        }'''
    stan_model = pystan.StanModel(model_code=schools_code)
    return stan_model.sampling(data=data,
                               iter=draws,
                               warmup=0,
                               chains=chains)


def pymc3_noncentered_schools(data, draws, chains):
    with pm.Model():
        mu = pm.Normal('mu', mu=0, sd=5)
        tau = pm.HalfCauchy('tau', beta=5)
        theta_tilde = pm.Normal('theta_tilde', mu=0, sd=1, shape=data['J'])
        theta = pm.Deterministic('theta', mu + tau * theta_tilde)
        pm.Normal('obs', mu=theta, sd=data['sigma'], observed=data['y'])
        return pm.sample(draws, chains=chains)


class CheckXarrayUtils(object):

    def check_varnames_coords_dims(self, varnames, coords, dims):
        expected_varnames = {'mu', 'tau', 'theta_tilde', 'theta'}
        assert set(varnames) == expected_varnames

        assert 'chain' in coords
        assert 'draw' in coords

        for varname in expected_varnames:
            assert varname in dims

    def test_default_varnames(self):
        varnames, coords, dims = self.cls.default_varnames_coords_dims(self.obj, None, None)
        self.check_varnames_coords_dims(varnames, coords, dims)

    def test_default_varnames_bad(self):
        varnames, coords, dims = self.cls.default_varnames_coords_dims(self.obj,
                                                                       {'a': range(2)},
                                                                       None)
        self.check_varnames_coords_dims(varnames, coords, dims)
        assert 'a' in coords

    def test_verify_coords_dims(self):
        varnames, coords, dims = self.cls.default_varnames_coords_dims(self.obj, None, None)
        self.check_varnames_coords_dims(varnames, coords, dims)

        # Both theta_tilde and theta need another coordinate
        good, message = self.cls(self.obj, coords, dims).verify_coords_dims()
        assert not good
        assert 'theta_tilde' in message
        assert 'theta' in message

    def test_verify_coords_dims_good(self):
        varnames, coords, dims = self.cls.default_varnames_coords_dims(
            self.obj,
            {'school': np.arange(self.data['J'])},
            {'theta': ['school'], 'theta_tilde': ['school']}
        )
        self.check_varnames_coords_dims(varnames, coords, dims)

        # Both theta_tilde and theta need another coordinate
        good, _ = self.cls(self.obj, coords, dims).verify_coords_dims()
        assert good

    def test_to_xarray(self):
        data = self.cls(
            self.obj,
            {'school': np.arange(self.data['J'])},
            {'theta': ['school'], 'theta_tilde': ['school']}
        ).to_xarray()
        assert data.draw.shape == (self.draws,)
        assert data.chain.shape == (self.chains,)
        assert data.school.shape == (self.data['J'],)
        assert data.theta.shape == (self.chains, self.draws, self.data['J'])

    def test_convert_to_xarray(self):
        # This does not use the class, and should work for all converters
        data = convert_to_xarray(
            self.obj,
            {'school': np.arange(self.data['J'])},
            {'theta': ['school'], 'theta_tilde': ['school']}
        )
        assert data.draw.shape == (self.draws,)
        assert data.chain.shape == (self.chains,)
        assert data.school.shape == (self.data['J'],)
        assert data.theta.shape == (self.chains, self.draws, self.data['J'])


    #def test_pymc3_to_xarray_bad(self):
    #    with pytest.raises(TypeError):
    #        pymc3_to_xarray(self.trace, None, None)


class TestPyMC3XarrayUtils(CheckXarrayUtils):

    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.data, cls.draws, cls.chains = eight_schools_params()
        cls.obj = pymc3_noncentered_schools(cls.data, cls.draws, cls.chains)
        cls.cls = PyMC3ToXarray


class TestPyStanXarrayUtils(CheckXarrayUtils):

    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.data, cls.draws, cls.chains = eight_schools_params()
        cls.obj = pystan_noncentered_schools(cls.data, cls.draws, cls.chains)
        cls.cls = PyStanToXarray
