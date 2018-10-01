# pylint: disable=no-member,invalid-name
import os
import numpy as np
import pymc3 as pm
import pytest

from arviz import (
    convert_to_inference_data,
    convert_to_dataset,
    from_cmdstan,
    from_pymc3,
    from_pystan,
    from_emcee,
)
from .helpers import eight_schools_params, load_cached_models, BaseArvizTest


class TestNumpyToDataArray():
    def test_1d_dataset(self):
        size = 100
        dataset = convert_to_dataset(np.random.randn(size))
        assert len(dataset.data_vars) == 1

        assert set(dataset.coords) == {'chain', 'draw'}
        assert dataset.chain.shape == (1,)
        assert dataset.draw.shape == (size,)

    def test_warns_bad_shape(self):
        # Shape should be (chain, draw, *shape)
        with pytest.warns(SyntaxWarning):
            convert_to_dataset(np.random.randn(100, 4))

    def test_nd_to_dataset(self):
        shape = (1, 2, 3, 4, 5)
        dataset = convert_to_dataset(np.random.randn(*shape))
        assert len(dataset.data_vars) == 1
        var_name = list(dataset.data_vars)[0]

        assert len(dataset.coords) == len(shape)
        assert dataset.chain.shape == shape[:1]
        assert dataset.draw.shape == shape[1:2]
        assert dataset[var_name].shape == shape

    def test_nd_to_inference_data(self):
        shape = (1, 2, 3, 4, 5)
        inference_data = convert_to_inference_data(np.random.randn(*shape), group='foo')
        assert hasattr(inference_data, 'foo')
        assert len(inference_data.foo.data_vars) == 1
        var_name = list(inference_data.foo.data_vars)[0]

        assert len(inference_data.foo.coords) == len(shape)
        assert inference_data.foo.chain.shape == shape[:1]
        assert inference_data.foo.draw.shape == shape[1:2]
        assert inference_data.foo[var_name].shape == shape


class TestConvertToDataset():
    def setup_method(self):
        # pylint: disable=attribute-defined-outside-init
        self.datadict = {
            'a': np.random.randn(100),
            'b': np.random.randn(1, 100, 10),
            'c': np.random.randn(1, 100, 3, 4),
        }
        self.coords = {'c1' : np.arange(3), 'c2' : np.arange(4), 'b1' : np.arange(10)}
        self.dims = {'b' : ['b1'], 'c' : ['c1', 'c2']}

    def test_use_all(self):
        dataset = convert_to_dataset(self.datadict,
                                     coords=self.coords,
                                     dims=self.dims)
        assert set(dataset.data_vars) == {'a', 'b', 'c'}
        assert set(dataset.coords) == {'chain', 'draw', 'c1', 'c2', 'b1'}

        assert set(dataset.a.coords) == {'chain', 'draw'}
        assert set(dataset.b.coords) == {'chain', 'draw', 'b1'}
        assert set(dataset.c.coords) == {'chain', 'draw', 'c1', 'c2'}

    def test_missing_coords(self):
        dataset = convert_to_dataset(self.datadict,
                                     coords=None,
                                     dims=self.dims)
        assert set(dataset.data_vars) == {'a', 'b', 'c'}
        assert set(dataset.coords) == {'chain', 'draw', 'c1', 'c2', 'b1'}

        assert set(dataset.a.coords) == {'chain', 'draw'}
        assert set(dataset.b.coords) == {'chain', 'draw', 'b1'}
        assert set(dataset.c.coords) == {'chain', 'draw', 'c1', 'c2'}

    def test_missing_dims(self):
        # missing dims
        coords = {
            'c_dim_0' : np.arange(3),
            'c_dim_1' : np.arange(4),
            'b_dim_0' : np.arange(10)
        }
        dataset = convert_to_dataset(self.datadict,
                                     coords=coords,
                                     dims=None)
        assert set(dataset.data_vars) == {'a', 'b', 'c'}
        assert set(dataset.coords) == {'chain', 'draw', 'c_dim_0', 'c_dim_1', 'b_dim_0'}

        assert set(dataset.a.coords) == {'chain', 'draw'}
        assert set(dataset.b.coords) == {'chain', 'draw', 'b_dim_0'}
        assert set(dataset.c.coords) == {'chain', 'draw', 'c_dim_0', 'c_dim_1'}

    def test_skip_dim_0(self):
        dims = {'c' : [None, 'c2']}
        coords = {
            'c_dim_0' : np.arange(3),
            'c2' : np.arange(4),
            'b_dim_0' : np.arange(10)
        }
        dataset = convert_to_dataset(self.datadict,
                                     coords=coords,
                                     dims=dims)
        assert set(dataset.data_vars) == {'a', 'b', 'c'}
        assert set(dataset.coords) == {'chain', 'draw', 'c_dim_0', 'c2', 'b_dim_0'}

        assert set(dataset.a.coords) == {'chain', 'draw'}
        assert set(dataset.b.coords) == {'chain', 'draw', 'b_dim_0'}
        assert set(dataset.c.coords) == {'chain', 'draw', 'c_dim_0', 'c2'}


def test_dict_to_dataset():
    datadict = {
        'a': np.random.randn(100),
        'b': np.random.randn(1, 100, 10)
    }
    dataset = convert_to_dataset(datadict,
                                 coords={'c': np.arange(10)},
                                 dims={'b': ['c']})
    assert set(dataset.data_vars) == {'a', 'b'}
    assert set(dataset.coords) == {'chain', 'draw', 'c'}

    assert set(dataset.a.coords) == {'chain', 'draw'}
    assert set(dataset.b.coords) == {'chain', 'draw', 'c'}


def test_convert_to_dataset_idempotent():
    first = convert_to_dataset(np.random.randn(100))
    second = convert_to_dataset(first)
    assert first.equals(second)


def test_convert_to_inference_data_idempotent():
    first = convert_to_inference_data(np.random.randn(100), group='foo')
    second = convert_to_inference_data(first)
    assert first.foo is second.foo


def test_convert_to_inference_data_from_file(tmpdir):
    first = convert_to_inference_data(np.random.randn(100), group='foo')
    filename = str(tmpdir.join('test_file.nc'))
    first.to_netcdf(filename)
    second = convert_to_inference_data(filename)
    assert first.foo.equals(second.foo)


def test_convert_to_inference_data_bad():
    with pytest.raises(ValueError):
        convert_to_inference_data(1)


def test_convert_to_dataset_bad(tmpdir):
    first = convert_to_inference_data(np.random.randn(100), group='foo')
    filename = str(tmpdir.join('test_file.nc'))
    first.to_netcdf(filename)
    with pytest.raises(ValueError):
        convert_to_dataset(filename, group='bar')


class TestDictNetCDFUtils(BaseArvizTest):

    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        _, stan_fit = load_cached_models(cls.draws, cls.chains)['pystan']
        stan_dict = stan_fit.extract(stan_fit.model_pars, permuted=False)
        cls.obj = {}
        for name, vals in stan_dict.items():
            if name not in {'y_hat', 'log_lik'}:  # extra vars
                cls.obj[name] = np.swapaxes(vals, 0, 1)

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {'mu', 'tau', 'theta_tilde', 'theta'}

        assert set(dataset.coords) == {'chain', 'draw', 'school'}

    def test_convert_to_inference_data(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, 'posterior')
        self.check_var_names_coords_dims(inference_data.posterior)

    def test_convert_to_dataset(self):
        data = convert_to_dataset(self.obj, group='posterior',
                                  coords={'school': np.arange(self.data['J'])},
                                  dims={'theta': ['school'], 'theta_tilde': ['school']},
                                 )
        assert data.draw.shape == (self.draws,)
        assert data.chain.shape == (self.chains,)
        assert data.school.shape == (self.data['J'],)
        assert data.theta.shape == (self.chains, self.draws, self.data['J'])

    def get_inference_data(self):
        return convert_to_inference_data(
            self.obj,
            group='posterior',
            coords={'school': np.arange(self.data['J'])},
            dims={'theta': ['school'], 'theta_tilde': ['school']},
        )

class TestEmceeNetCDFUtils(BaseArvizTest):

    @classmethod
    def setup_class(cls):
        cls.draws, cls.chains = 500, 20
        fake_chains = cls.chains // 10  # emcee uses lots of walkers
        cls.obj = load_cached_models(cls.draws, fake_chains)['emcee']

    def get_inference_data(self):

        return from_emcee(
            self.obj,
            var_names=['ln(f)', 'b', 'm']
        )

    def test__verify_var_names(self):
        with pytest.raises(ValueError):
            from_emcee(self.obj, var_names=['not', 'enough'])

    def test__verify_arg_names(self):
        with pytest.raises(ValueError):
            from_emcee(self.obj, arg_names=['not', 'enough'])



class TestPyMC3NetCDFUtils(BaseArvizTest):

    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        cls.model, cls.obj = load_cached_models(cls.draws, cls.chains)['pymc3']

    def get_inference_data(self):
        with self.model:
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(self.obj)

        return from_pymc3(
            trace=self.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={'school': np.arange(self.data['J'])},
            dims={'theta': ['school'], 'theta_tilde': ['school']},
        )

    def test_sampler_stats(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, 'sample_stats')

    def test_posterior_predictive(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, 'posterior_predictive')

    def test_prior(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, 'prior')


class TestPyStanNetCDFUtils(BaseArvizTest):

    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        cls.model, cls.obj = load_cached_models(cls.draws, cls.chains)['pystan']

    def get_inference_data(self):
        return from_pystan(fit=self.obj,
                           posterior_predictive='y_hat',
                           observed_data=['y'],
                           log_likelihood='log_lik',
                           coords={'school': np.arange(self.data['J'])},
                           dims={'theta': ['school'],
                                 'y': ['school'],
                                 'log_lik': ['school'],
                                 'y_hat': ['school'],
                                 'theta_tilde': ['school']
                                }
                          )

    def get_inference_data2(self):
        # dictionary
        observed_data = {'y_hat' : self.data['y']}
        # ndarray
        log_likelihood = self.obj.extract('log_lik', permuted=False)['log_lik']
        return from_pystan(fit=self.obj,
                           posterior_predictive='y_hat',
                           observed_data=observed_data,
                           log_likelihood=log_likelihood,
                           coords={'school': np.arange(self.data['J'])},
                           dims={'theta': ['school'],
                                 'y': ['school'],
                                 'log_lik': ['school'],
                                 'y_hat': ['school'],
                                 'theta_tilde': ['school']
                                }
                          )

    def test_sampler_stats(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, 'sample_stats')

    def test_inference_data(self):
        inference_data1 = self.get_inference_data()
        inference_data2 = self.get_inference_data2()
        assert hasattr(inference_data1.sample_stats, 'log_likelihood')
        assert hasattr(inference_data1.observed_data, 'y')
        assert hasattr(inference_data2.sample_stats, 'log_likelihood')
        assert hasattr(inference_data2.observed_data, 'y_hat')


class TestCmdStanNetCDFUtils(BaseArvizTest):

    @classmethod
    def setup_class(cls):
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, 'saved_models')
        cls.paths = {
            'no_warmup' : [
                os.path.join(data_directory, "cmdstan/output_no_warmup1.csv"),
                os.path.join(data_directory, "cmdstan/output_no_warmup2.csv"),
                os.path.join(data_directory, "cmdstan/output_no_warmup3.csv"),
                os.path.join(data_directory, "cmdstan/output_no_warmup4.csv"),
            ],
            'warmup' : [
                os.path.join(data_directory, "cmdstan/output_warmup1.csv"),
                os.path.join(data_directory, "cmdstan/output_warmup2.csv"),
                os.path.join(data_directory, "cmdstan/output_warmup3.csv"),
                os.path.join(data_directory, "cmdstan/output_warmup4.csv"),
            ],
            'no_warmup_glob' : os.path.join(data_directory,
                                            "cmdstan/output_no_warmup[0-9].csv"
                                           ),
            'warmup_glob' : os.path.join(data_directory,
                                         "cmdstan/output_warmup[0-9].csv"
                                        ),
            'combined_no_warmup' : [
                os.path.join(data_directory, "cmdstan/combined_output_no_warmup.csv")
            ],
            'combined_warmup' : [
                os.path.join(data_directory, "cmdstan/combined_output_warmup.csv")
            ],
            'combined_no_warmup_glob' : os.path.join(data_directory,
                                                     "cmdstan/combined_output_no_warmup.csv"
                                                    ),
            'combined_warmup_glob' : os.path.join(data_directory,
                                                  "cmdstan/combined_output_warmup.csv"
                                                 ),
            'eight_schools_glob' : os.path.join(data_directory,
                                                "cmdstan/eight_schools_output[0-9].csv"
                                               ),
            'eight_schools' : [
                os.path.join(data_directory, "cmdstan/eight_schools_output1.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output2.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output3.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output4.csv"),
            ],
            'missing_files' : [
                os.path.join(data_directory, "cmdstan/combined_missing_config.csv"),
                os.path.join(data_directory, "cmdstan/combined_missing_adaptation.csv"),
                os.path.join(data_directory, "cmdstan/combined_missing_timing1.csv"),
                os.path.join(data_directory, "cmdstan/combined_missing_timing2.csv"),
            ],
        }
        cls.observed_data_paths = [
            os.path.join(data_directory, "cmdstan/eight_schools.data.R"),
            os.path.join(data_directory, "cmdstan/example_stan.data.R"),
        ]

    def get_inference_data(self, output, **kwargs):
        return from_cmdstan(output=output, **kwargs)

    def test_sample_stats(self):
        for key, path in self.paths.items():
            if 'missing' in key:
                continue
            inference_data = self.get_inference_data(path)
            assert hasattr(inference_data, 'sample_stats')

    def test_inference_data_shapes(self):
        """Assert that shapes are transformed correctly"""
        for key, path in self.paths.items():
            if 'eight' in key or 'missing' in key:
                continue
            inference_data = self.get_inference_data(path)
            assert hasattr(inference_data, 'posterior')
            assert hasattr(inference_data.posterior, 'y')
            assert hasattr(inference_data.posterior, 'x')
            assert hasattr(inference_data.posterior, 'Z')
            assert inference_data.posterior['y'].shape == (4, 100)
            assert inference_data.posterior['x'].shape == (4, 100, 3)
            assert inference_data.posterior['Z'].shape == (4, 100, 4, 6)
            dims = ['chain', 'draw']
            y_mean_true = 0
            y_mean = inference_data.posterior['y'].mean(dim=dims)
            assert np.isclose(y_mean, y_mean_true, atol=1e-1)
            x_mean_true = np.array([1, 2, 3])
            x_mean = inference_data.posterior['x'].mean(dim=dims)
            assert np.isclose(x_mean, x_mean_true, atol=1e-1).all()
            Z_mean_true = np.array([1, 2, 3, 4])
            Z_mean = inference_data.posterior['Z'].mean(dim=dims).mean(axis=1)
            assert np.isclose(Z_mean, Z_mean_true, atol=7e-1).all()

    def test_inference_data_input_types1(self):
        """Check input types

            output --> str, list of str
            prior --> str, list of str
            posterior_predictive --> str, variable in output
            observed_data --> Rdump format
            observed_data_var --> str, variable
            log_likelihood --> str
            coords --> one to many
            dims --> one to many
        """
        for key, path in self.paths.items():
            if 'eight' not in key:
                continue
            inference_data = self.get_inference_data(
                output=path,
                prior=path,
                posterior_predictive='y_hat',
                observed_data=self.observed_data_paths[0],
                observed_data_var='y',
                log_likelihood='log_lik',
                coords={'school': np.arange(8)},
                dims={'theta': ['school'],
                      'y': ['school'],
                      'log_lik': ['school'],
                      'y_hat': ['school'],
                      'theta_tilde': ['school'],
                     }
            )
            assert hasattr(inference_data, 'posterior')
            assert hasattr(inference_data, 'sample_stats')
            assert hasattr(inference_data.sample_stats, 'log_likelihood')
            assert hasattr(inference_data, 'posterior_predictive')
            assert hasattr(inference_data, 'observed_data')

    def test_inference_data_input_types2(self):
        """Check input types (change, see earlier)

            posterior_predictive --> List[str], variable in output
            observed_data_var --> List[str], variable
        """
        for key, path in self.paths.items():
            if 'eight' not in key:
                continue
            inference_data = self.get_inference_data(
                output=path,
                prior=path,
                posterior_predictive=['y_hat'],
                observed_data=self.observed_data_paths[0],
                observed_data_var=['y'],
                log_likelihood='log_lik',
                coords={'school': np.arange(8)},
                dims={'theta': ['school'],
                      'y': ['school'],
                      'log_lik': ['school'],
                      'y_hat': ['school'],
                      'theta_tilde': ['school']
                     }
            )
            assert hasattr(inference_data, 'posterior')
            assert hasattr(inference_data, 'sample_stats')
            assert hasattr(inference_data.sample_stats, 'log_likelihood')
            assert hasattr(inference_data, 'posterior_predictive')
            assert hasattr(inference_data, 'observed_data')

    def test_inference_data_input_types3(self):
        """Check input types (change, see earlier)

            posterior_predictive --> str, csv file
            coords --> one to many + one to one (default dim)
            dims --> one to many
        """
        for key, path in self.paths.items():
            if 'eight' not in key:
                continue
            post_pred = self.paths['eight_schools_glob']
            inference_data = self.get_inference_data(
                output=path,
                prior=path,
                posterior_predictive=post_pred,
                observed_data=self.observed_data_paths[0],
                observed_data_var=['y'],
                log_likelihood='log_lik',
                coords={'school': np.arange(8), 'log_lik_dim_0' : np.arange(8)},
                dims={'theta': ['school'],
                      'y': ['school'],
                      'y_hat': ['school'],
                      'theta_tilde': ['school']
                     }
            )
            assert hasattr(inference_data, 'posterior')
            assert hasattr(inference_data, 'sample_stats')
            assert hasattr(inference_data.sample_stats, 'log_likelihood')
            assert hasattr(inference_data, 'posterior_predictive')
            assert hasattr(inference_data, 'observed_data')

    def test_inference_data_input_types4(self):
        """Check input types (change, see earlier)

            coords --> one to many + one to one (non-default dim)
            dims --> one to many + one to one
        """
        for key, path in self.paths.items():
            if 'eight' not in key:
                continue
            post_pred = self.paths['eight_schools']
            inference_data = self.get_inference_data(
                output=path,
                prior=path,
                posterior_predictive=post_pred,
                observed_data=self.observed_data_paths[0],
                observed_data_var=['y'],
                log_likelihood=['log_lik'],
                coords={'school': np.arange(8),
                        'log_lik_dim' : np.arange(8)},
                dims={'theta': ['school'],
                      'y': ['school'],
                      'log_lik': ['log_lik_dim'],
                      'y_hat': ['school'],
                      'theta_tilde': ['school']
                     }
            )
            assert hasattr(inference_data, 'posterior')
            assert hasattr(inference_data, 'sample_stats')
            assert hasattr(inference_data.sample_stats, 'log_likelihood')
            assert hasattr(inference_data, 'posterior_predictive')
            assert hasattr(inference_data, 'observed_data')

    def test_inference_data_bad_csv(self):
        """Check ValueError for csv with missing headers"""
        for key, paths in self.paths.items():
            if 'missing' not in key:
                continue
            for path in paths:
                with pytest.raises(ValueError):
                    self.get_inference_data(output=path)

    def test_inference_data_observed_data1(self):
        """Read Rdump, check shapes are correct

            All variables
        """
        path = self.observed_data_paths[1]
        inference_data = self.get_inference_data(
            output=None,
            observed_data=path,
        )
        assert hasattr(inference_data, 'observed_data')
        assert len(inference_data.observed_data.data_vars) == 3
        assert inference_data.observed_data['x'].shape == (1,)
        assert inference_data.observed_data['y'].shape == (3,)
        assert inference_data.observed_data['Z'].shape == (4, 5)

    def test_inference_data_observed_data2(self):
        """Read Rdump, check shapes are correct

            One variable as str
        """
        path = self.observed_data_paths[1]
        inference_data = self.get_inference_data(
            output=None,
            observed_data=path,
            observed_data_var='x',
        )
        assert hasattr(inference_data, 'observed_data')
        assert len(inference_data.observed_data.data_vars) == 1
        assert inference_data.observed_data['x'].shape == (1,)

    def test_inference_data_observed_data3(self):
        """Read Rdump, check shapes are correct

            One variable as a list
        """
        path = self.observed_data_paths[1]
        inference_data = self.get_inference_data(
            output=None,
            observed_data=path,
            observed_data_var=['x'],
        )
        assert hasattr(inference_data, 'observed_data')
        assert len(inference_data.observed_data.data_vars) == 1
        assert inference_data.observed_data['x'].shape == (1,)

    def test_inference_data_observed_data4(self):
        """Read Rdump, check shapes are correct

            Many variables as list
        """
        path = self.observed_data_paths[1]
        inference_data = self.get_inference_data(
            output=None,
            observed_data=path,
            observed_data_var=['y', 'Z'],
        )
        assert hasattr(inference_data, 'observed_data')
        assert len(inference_data.observed_data.data_vars) == 2
        assert inference_data.observed_data['y'].shape == (3,)
        assert inference_data.observed_data['Z'].shape == (4, 5)
