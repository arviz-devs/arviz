# pylint: disable=no-member
import numpy as np
#import pytest

from .helpers import eight_schools_params, load_cached_models, BaseArvizTest
from ..utils.xarray_utils import convert_to_netcdf, get_converter, \
                                 DictToNetCDF, PyMC3ToNetCDF, PyStanToNetCDF


class CheckNetCDFUtils(BaseArvizTest):

    def check_varnames_coords_dims(self, varnames, coords, dims):
        expected_varnames = {'mu', 'tau', 'theta_tilde', 'theta'}
        assert set(varnames) == expected_varnames

        assert 'chain' in coords
        assert 'draw' in coords

        for varname in expected_varnames:
            assert varname in dims

    def test_default_varnames(self):
        converter = get_converter(self.obj, coords=None, dims=None, chains=self.chains)
        self.check_varnames_coords_dims(converter.varnames, converter.coords, converter.dims)

    def test_default_varnames_bad(self):
        converter = get_converter(self.obj, coords={'a': range(2)}, dims=None, chains=self.chains)
        self.check_varnames_coords_dims(converter.varnames, converter.coords, converter.dims)
        assert 'a' in converter.coords

    def test_verify_coords_dims(self):
        converter = get_converter(self.obj, coords=None, dims=None, chains=self.chains)
        self.check_varnames_coords_dims(converter.varnames, converter.coords, converter.dims)

        # Both theta_tilde and theta need another coordinate
        good, message = converter.verify_coords_dims()
        assert not good
        assert 'theta_tilde' in message
        assert 'theta' in message

    def test_to_netcdf(self):
        data = get_converter(
            self.obj,
            coords={'school': np.arange(self.data['J'])},
            dims={'theta': ['school'], 'theta_tilde': ['school']},
            chains=self.chains
        ).to_netcdf()

        assert data.posterior.draw.shape == (self.draws,)
        assert data.posterior.chain.shape == (self.chains,)
        assert data.posterior.school.shape == (self.data['J'],)
        assert data.posterior.theta.shape == (self.chains, self.draws, self.data['J'])

    def test_convert_to_netcdf(self):
        # This does not use the class, and should work for all converters
        data = convert_to_netcdf(
            self.obj,
            coords={'school': np.arange(self.data['J'])},
            dims={'theta': ['school'], 'theta_tilde': ['school']},
            chains=self.chains
        )
        assert data.posterior.draw.shape == (self.draws,)
        assert data.posterior.chain.shape == (self.chains,)
        assert data.posterior.school.shape == (self.data['J'],)
        assert data.posterior.theta.shape == (self.chains, self.draws, self.data['J'])


class TestDictNetCDFUtils(CheckNetCDFUtils):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        _, stan_fit = load_cached_models(cls.draws, cls.chains)['pystan']
        cls.obj = stan_fit.extract(stan_fit.model_pars, permuted=False)
        cls.cls = DictToNetCDF


class TestPyMC3NetCDFUtils(CheckNetCDFUtils):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        cls.model, cls.obj = load_cached_models(cls.draws, cls.chains)['pymc3']
        cls.cls = PyMC3ToNetCDF


class TestPyStanNetCDFUtils(CheckNetCDFUtils):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        cls.model, cls.obj = load_cached_models(cls.draws, cls.chains)['pystan']
        cls.cls = PyStanToNetCDF
