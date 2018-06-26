# pylint: disable=no-member
import numpy as np
#import pytest

from .helpers import eight_schools_params, load_cached_models
from ..utils.xarray_utils import convert_to_xarray, PyMC3ToXarray, PyStanToXarray


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
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        cls.model, cls.obj = load_cached_models(cls.draws, cls.chains)['pymc3']
        cls.cls = PyMC3ToXarray


class TestPyStanXarrayUtils(CheckXarrayUtils):

    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        cls.model, cls.obj = load_cached_models(cls.draws, cls.chains)['pystan']
        cls.cls = PyStanToXarray
