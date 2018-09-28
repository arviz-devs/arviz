# pylint: disable=redefined-outer-name
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
import pymc3 as pm
import pytest

from arviz import from_pymc3, compare
from .helpers import eight_schools_params, load_cached_models
from ..plots import (plot_density, plot_trace, plot_energy, plot_posterior,
                     plot_autocorr, plot_forest, plot_parallel, plot_pair,
                     plot_joint, plot_ppc, plot_violin, plot_compare)


np.random.seed(0)


@pytest.fixture(scope='module')
def models(request):
    class Models:
        models = load_cached_models(draws=500, chains=2)
        pymc3_model, pymc3_fit = models['pymc3']
        stan_model, stan_fit = models['pystan']

    def fin():
        plt.close('all')

    request.addfinalizer(fin)
    return Models()


@pytest.fixture(scope='module')
def pymc3_sample_ppc(models):

    with models.pymc3_model:
        return pm.sample_ppc(models.pymc3_fit, 100)


@pytest.fixture(scope='module')
def data():
    data = eight_schools_params()
    return data


@pytest.fixture(scope='module')
def df_trace():
    return DataFrame({'a': np.random.poisson(2.3, 100)})


@pytest.mark.parametrize("kwargs", [{"point_estimate": "mean"},
                                    {"point_estimate": "median"},
                                    {"outline": True},
                                    {"colors": ["g", "b"]},
                                    {"colors": "gb"},
                                    {"hpd_markers": ["v"]},
                                    {"shade": 1}])
def test_plot_density_float(models, kwargs):
    obj = [getattr(models, model_fit) for model_fit in ["pymc3_fit", "stan_fit"]]
    axes = plot_density(obj, **kwargs)
    assert axes.shape[0] >= 18
    assert axes.shape[1] == 1


def test_plot_density_int():
    axes = plot_density(np.random.randint(10, size=10), shade=.9)
    assert axes.shape[1] == 1


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize('combined', [True, False])
def test_plot_trace(models, combined, model_fit):
    obj = getattr(models, model_fit)
    axes = plot_trace(obj, var_names=('mu', 'tau'), combined=combined, lines=[('mu', {}, [1, 2])])
    assert axes.shape == (2, 2)


@pytest.mark.parametrize("model_fits", [["pymc3_fit"], ["stan_fit"], ["pymc3_fit", "stan_fit"]])
@pytest.mark.parametrize("args_expected", [(dict(), (3,)),
                                           (dict(r_hat=False, quartiles=False), (2,)),
                                           (dict(var_names=['mu'], colors='C0', n_eff=False,
                                                 combined=True), (2,)),
                                           (dict(kind='ridgeplot', r_hat=False, n_eff=False), (1,))
                                           ])
def test_plot_forest(models, model_fits, args_expected):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    args, expected = args_expected
    _, axes = plot_forest(obj, **args)
    assert axes.shape == expected


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize('kind', ['kde', 'hist'])
def test_plot_energy(models, model_fit, kind):
    obj = getattr(models, model_fit)
    assert plot_energy(obj, kind=kind)


def test_plot_parallel_raises_valueerror(df_trace):  # pylint: disable=invalid-name
    with pytest.raises(ValueError):
        plot_parallel(df_trace)


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_parallel_exception(models, model_fit):
    obj = getattr(models, model_fit)
    assert plot_parallel(obj)


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize('kind', ['scatter', 'hexbin'])
def test_plot_joint(models, model_fit, kind):
    obj = getattr(models, model_fit)
    plot_joint(obj, var_names=('mu', 'tau'), kind=kind)


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize("kwargs", [{"var_names":['theta'], "divergences":True,
                                     "coords":{'theta_dim_0': [0, 1]},
                                     "plot_kwargs":{'marker': 'x'},
                                     "divergences_kwargs": {'marker': '*', 'c': 'C'}},

                                    {"divergences":True, "plot_kwargs":{'marker': 'x'},
                                     "divergences_kwargs": {'marker': '*', 'c': 'C'}},

                                    {"kind":'hexbin', "var_names": ['theta'],
                                     "coords":{'theta_dim_0': [0, 1]},
                                     "plot_kwargs":{'cmap': 'viridis'}, "textsize": 20}])
def test_plot_pair(models, model_fit, kwargs):
    obj = getattr(models, model_fit)
    ax, _ = plot_pair(obj, **kwargs)
    assert ax


@pytest.mark.parametrize('kind', ['density', 'cumulative'])
def test_plot_ppc(models, pymc3_sample_ppc, kind):
    data = from_pymc3(trace=models.pymc3_fit,
                      posterior_predictive=pymc3_sample_ppc)
    axes = plot_ppc(data, kind=kind)
    assert axes


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_violin(models, model_fit):
    obj = getattr(models, model_fit)
    axes = plot_violin(obj)
    assert axes.shape[0] >= 18


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_autocorr_uncombined(models, model_fit):
    obj = getattr(models, model_fit)
    axes = plot_autocorr(obj, combined=False)
    assert axes.shape[0] == 1
    assert (axes.shape[1] == 36 and model_fit == "pymc3_fit" or
            axes.shape[1] == 68 and model_fit == "stan_fit")


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_autocorr_combined(models, model_fit):
    obj = getattr(models, model_fit)
    axes = plot_autocorr(obj, combined=True)
    assert axes.shape[0] == 1
    assert (axes.shape[1] == 18 and model_fit == "pymc3_fit" or
            axes.shape[1] == 34 and model_fit == "stan_fit")


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_posterior(models, model_fit):
    obj = getattr(models, model_fit)
    axes = plot_posterior(obj, var_names=('mu', 'tau'), rope=(-2, 2), ref_val=0)
    assert axes.shape == (2,)


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize("point_estimate", ('mode', 'mean', 'median'))
def test_point_estimates(models, model_fit, point_estimate):
    obj = getattr(models, model_fit)
    axes = plot_posterior(obj, var_names=('mu', 'tau'), point_estimate=point_estimate)
    assert axes.shape == (2,)


@pytest.mark.parametrize("kwargs", [{"insample_dev": False},
                                    {"plot_standard_error": False},
                                    {"plot_ic_diff": False}
                                    ])
def test_plot_compare(models, kwargs):

    # Pymc3 models create loglikelihood on InferenceData automatically
    model_compare = compare({
        'Pymc3': models.pymc3_fit,
        'Pymc3_Again': models.pymc3_fit,
    })

    axes = plot_compare(model_compare, **kwargs)
    assert axes
