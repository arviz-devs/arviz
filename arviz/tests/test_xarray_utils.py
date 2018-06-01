import numpy as np
import pytest

from ..compat import pymc3 as pm
from ..utils.xarray_utils import pymc3_to_xarray, default_varnames_coords_dims, verify_coords_dims


class TestXarrayUtils(object):

    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.J = 8
        y = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
        sigma = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
        cls.draws = 500
        cls.chains = 2
        with pm.Model():
            mu = pm.Normal('mu', mu=0, sd=5)
            tau = pm.HalfCauchy('tau', beta=5)
            theta_tilde = pm.Normal('theta_tilde', mu=0, sd=1, shape=cls.J)
            theta = pm.Deterministic('theta', mu + tau * theta_tilde)
            pm.Normal('obs', mu=theta, sd=sigma, observed=y)
            cls.trace = pm.sample(cls.draws, chains=cls.chains)

    def check_varnames_coords_dims(self, varnames, coords, dims):
        expected_varnames = {'mu', 'tau', 'theta_tilde', 'theta'}
        assert set(varnames) == expected_varnames

        assert 'chain' in coords
        assert 'draw' in coords

        for varname in expected_varnames:
            assert varname in dims

    def test_default_varnames(self):
        varnames, coords, dims = default_varnames_coords_dims(self.trace, None, None)
        self.check_varnames_coords_dims(varnames, coords, dims)

    def test_default_varnames_bad(self):
        varnames, coords, dims = default_varnames_coords_dims(self.trace, {'a': range(2)}, None)
        self.check_varnames_coords_dims(varnames, coords, dims)
        assert 'a' in coords

    def test_verify_coords_dims(self):
        varnames, coords, dims = default_varnames_coords_dims(self.trace, None, None)
        self.check_varnames_coords_dims(varnames, coords, dims)

        # Both theta_tilde and theta need another coordinate
        good, message = verify_coords_dims(varnames, self.trace, coords, dims)
        assert not good
        assert 'theta_tilde' in message
        assert 'theta' in message

    def test_verify_coords_dims_good(self):
        varnames, coords, dims = default_varnames_coords_dims(
            self.trace,
            {'school': np.arange(self.J)},
            {'theta': ['school'], 'theta_tilde': ['school']}
        )
        self.check_varnames_coords_dims(varnames, coords, dims)

        # Both theta_tilde and theta need another coordinate
        good, _ = verify_coords_dims(varnames, self.trace, coords, dims)
        assert good

    def test_pymc3_to_xarray(self):
        data = pymc3_to_xarray(
            self.trace,
            {'school': np.arange(self.J)},
            {'theta': ['school'], 'theta_tilde': ['school']}
        )
        assert data.draw.shape == (self.draws,)
        assert data.chain.shape == (self.chains,)
        assert data.school.shape == (self.J,)
        assert data.theta.shape == (self.chains, self.draws, self.J)

    def test_pymc3_to_xarray_bad(self):
        with pytest.raises(TypeError):
            pymc3_to_xarray(self.trace, None, None)
