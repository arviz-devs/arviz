# pylint: disable=redefined-outer-name
import numpy as np
import xarray as xr
import pytest

from ..data import from_dict
from ..plots.plot_utils import (
    make_2d,
    xarray_to_ndarray,
    xarray_var_iter,
    get_bins,
    get_coords,
    filter_plotters_list,
    format_sig_figs,
    get_plotting_function,
)
from ..rcparams import rc_context


@pytest.mark.parametrize(
    "value, default, expected",
    [
        (123.456, 2, 3),
        (-123.456, 3, 3),
        (-123.456, 4, 4),
        (12.3456, 2, 2),
        (1.23456, 2, 2),
        (0.123456, 2, 2),
    ],
)
def test_format_sig_figs(value, default, expected):
    assert format_sig_figs(value, default=default) == expected


@pytest.fixture(scope="function")
def sample_dataset():
    mu = np.arange(1, 7).reshape(2, 3)
    tau = np.arange(7, 13).reshape(2, 3)

    chain = [0, 1]
    draws = [0, 1, 2]

    data = xr.Dataset(
        {"mu": (["chain", "draw"], mu), "tau": (["chain", "draw"], tau)},
        coords={"draw": draws, "chain": chain},
    )

    return mu, tau, data


def test_make_2d():
    """Touches code that is hard to reach."""
    assert len(make_2d(np.array([2, 3, 4])).shape) == 2


def test_get_bins():
    """Touches code that is hard to reach."""
    assert get_bins(np.array([1, 2, 3, 100])) is not None


def test_dataset_to_numpy_not_combined(sample_dataset):  # pylint: disable=invalid-name
    mu, tau, data = sample_dataset
    var_names, data = xarray_to_ndarray(data, combined=False)

    # 2 vars x 2 chains
    assert len(var_names) == 4
    mu_tau = np.concatenate((mu, tau), axis=0)
    tau_mu = np.concatenate((tau, mu), axis=0)
    deqmt = data == mu_tau
    deqtm = data == tau_mu
    assert deqmt.all() or deqtm.all()


def test_dataset_to_numpy_combined(sample_dataset):
    mu, tau, data = sample_dataset
    var_names, data = xarray_to_ndarray(data, combined=True)

    assert len(var_names) == 2
    assert (data[var_names.index("mu")] == mu.reshape(1, 6)).all()
    assert (data[var_names.index("tau")] == tau.reshape(1, 6)).all()


def test_xarray_var_iter_ordering():
    """Assert that coordinate names stay the provided order"""
    coords = list("dcba")
    data = from_dict(  # pylint: disable=no-member
        {"x": np.random.randn(1, 100, len(coords))},
        coords={"in_order": coords},
        dims={"x": ["in_order"]},
    ).posterior

    coord_names = [sel["in_order"] for _, sel, _ in xarray_var_iter(data)]
    assert coord_names == coords


def test_xarray_var_iter_ordering_combined(sample_dataset):  # pylint: disable=invalid-name
    """Assert that varname order stays consistent when chains are combined"""
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_var_iter(data, var_names=None, combined=True)]
    assert set(var_names) == {"mu", "tau"}


def test_xarray_var_iter_ordering_uncombined(sample_dataset):  # pylint: disable=invalid-name
    """Assert that varname order stays consistent when chains are not combined"""
    _, _, data = sample_dataset
    var_names = [(var, selection) for (var, selection, _) in xarray_var_iter(data, var_names=None)]

    assert len(var_names) == 4
    for var_name in var_names:
        assert var_name in [
            ("mu", {"chain": 0}),
            ("mu", {"chain": 1}),
            ("tau", {"chain": 0}),
            ("tau", {"chain": 1}),
        ]


def test_xarray_var_data_array(sample_dataset):  # pylint: disable=invalid-name
    """Assert that varname order stays consistent when chains are combined

    Touches code that is hard to reach.
    """
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_var_iter(data.mu, var_names=None, combined=True)]
    assert set(var_names) == {"mu"}


class TestCoordsExceptions:
    # test coord exceptions on datasets
    def test_invalid_coord_name(self, sample_dataset):  # pylint: disable=invalid-name
        """Assert that nicer exception appears when user enters wrong coords name"""
        _, _, data = sample_dataset
        coords = {"NOT_A_COORD_NAME": [1]}

        with pytest.raises(
            ValueError, match="Coords {'NOT_A_COORD_NAME'} are invalid coordinate keys"
        ):
            get_coords(data, coords)

    def test_invalid_coord_value(self, sample_dataset):  # pylint: disable=invalid-name
        """Assert that nicer exception appears when user enters wrong coords value"""
        _, _, data = sample_dataset
        coords = {"draw": [1234567]}

        with pytest.raises(
            KeyError, match=r"Coords should follow mapping format {coord_name:\[dim1, dim2\]}"
        ):
            get_coords(data, coords)

    def test_invalid_coord_structure(self, sample_dataset):  # pylint: disable=invalid-name
        """Assert that nicer exception appears when user enters wrong coords datatype"""
        _, _, data = sample_dataset
        coords = {"draw"}

        with pytest.raises(TypeError):
            get_coords(data, coords)

    # test coord exceptions on dataset list
    def test_invalid_coord_name_list(self, sample_dataset):  # pylint: disable=invalid-name
        """Assert that nicer exception appears when user enters wrong coords name"""
        _, _, data = sample_dataset
        coords = {"NOT_A_COORD_NAME": [1]}

        with pytest.raises(
            ValueError, match=r"data\[1\]:.+Coords {'NOT_A_COORD_NAME'} are invalid coordinate keys"
        ):
            get_coords((data, data), ({"draw": [0, 1]}, coords))

    def test_invalid_coord_value_list(self, sample_dataset):  # pylint: disable=invalid-name
        """Assert that nicer exception appears when user enters wrong coords value"""
        _, _, data = sample_dataset
        coords = {"draw": [1234567]}

        with pytest.raises(
            KeyError,
            match=r"data\[0\]:.+Coords should follow mapping format {coord_name:\[dim1, dim2\]}",
        ):
            get_coords((data, data), (coords, {"draw": [0, 1]}))


def test_filter_plotter_list():
    plotters = list(range(7))
    with rc_context({"plot.max_subplots": 10}):
        plotters_filtered = filter_plotters_list(plotters, "")
    assert plotters == plotters_filtered


def test_filter_plotter_list_warning():
    plotters = list(range(7))
    with rc_context({"plot.max_subplots": 5}):
        with pytest.warns(SyntaxWarning, match="test warning"):
            plotters_filtered = filter_plotters_list(plotters, "test warning")
    assert len(plotters_filtered) == 5


def test_bokeh_import():
    """Tests that correct method is returned on bokeh import"""
    plot = get_plotting_function("plot_dist", "distplot", "bokeh")

    from arviz.plots.backends.bokeh.distplot import plot_dist

    assert plot is plot_dist
