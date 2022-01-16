# pylint: disable=redefined-outer-name
import importlib

import numpy as np
import pytest
import xarray as xr

from ...data import from_dict
from ...plots.backends.matplotlib import dealiase_sel_kwargs, matplotlib_kwarg_dealiaser
from ...plots.plot_utils import (
    compute_ranks,
    filter_plotters_list,
    format_sig_figs,
    get_plotting_function,
    make_2d,
    set_bokeh_circular_ticks_labels,
    vectorized_to_hex,
)
from ...rcparams import rc_context
from ...sel_utils import xarray_sel_iter, xarray_to_ndarray
from ...stats.density_utils import get_bins
from ...utils import get_coords
from ..helpers import running_on_ci

# Check if Bokeh is installed
bokeh_installed = importlib.util.find_spec("bokeh") is not None  # pylint: disable=invalid-name


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


def test_xarray_sel_iter_ordering():
    """Assert that coordinate names stay the provided order"""
    coords = list("dcba")
    data = from_dict(  # pylint: disable=no-member
        {"x": np.random.randn(1, 100, len(coords))},
        coords={"in_order": coords},
        dims={"x": ["in_order"]},
    ).posterior

    coord_names = [sel["in_order"] for _, sel, _ in xarray_sel_iter(data)]
    assert coord_names == coords


def test_xarray_sel_iter_ordering_combined(sample_dataset):  # pylint: disable=invalid-name
    """Assert that varname order stays consistent when chains are combined"""
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_sel_iter(data, var_names=None, combined=True)]
    assert set(var_names) == {"mu", "tau"}


def test_xarray_sel_iter_ordering_uncombined(sample_dataset):  # pylint: disable=invalid-name
    """Assert that varname order stays consistent when chains are not combined"""
    _, _, data = sample_dataset
    var_names = [(var, selection) for (var, selection, _) in xarray_sel_iter(data, var_names=None)]

    assert len(var_names) == 4
    for var_name in var_names:
        assert var_name in [
            ("mu", {"chain": 0}),
            ("mu", {"chain": 1}),
            ("tau", {"chain": 0}),
            ("tau", {"chain": 1}),
        ]


def test_xarray_sel_data_array(sample_dataset):  # pylint: disable=invalid-name
    """Assert that varname order stays consistent when chains are combined

    Touches code that is hard to reach.
    """
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_sel_iter(data.mu, var_names=None, combined=True)]
    assert set(var_names) == {"mu"}


class TestCoordsExceptions:
    # test coord exceptions on datasets
    def test_invalid_coord_name(self, sample_dataset):  # pylint: disable=invalid-name
        """Assert that nicer exception appears when user enters wrong coords name"""
        _, _, data = sample_dataset
        coords = {"NOT_A_COORD_NAME": [1]}

        with pytest.raises(
            (KeyError, ValueError),
            match=(
                r"Coords "
                r"({'NOT_A_COORD_NAME'} are invalid coordinate keys"
                r"|should follow mapping format {coord_name:\[dim1, dim2\]})"
            ),
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
            (KeyError, ValueError),
            match=(
                r"data\[1\]:.+Coords "
                r"({'NOT_A_COORD_NAME'} are invalid coordinate keys"
                r"|should follow mapping format {coord_name:\[dim1, dim2\]})"
            ),
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
        with pytest.warns(UserWarning, match="test warning"):
            plotters_filtered = filter_plotters_list(plotters, "test warning")
    assert len(plotters_filtered) == 5


@pytest.mark.skipif(
    not (bokeh_installed or running_on_ci()),
    reason="test requires bokeh which is not installed",
)
def test_bokeh_import():
    """Tests that correct method is returned on bokeh import"""
    plot = get_plotting_function("plot_dist", "distplot", "bokeh")

    from ...plots.backends.bokeh.distplot import plot_dist

    assert plot is plot_dist


@pytest.mark.parametrize(
    "params",
    [
        {
            "input": (
                {
                    "dashes": "-",
                },
                "scatter",
            ),
            "output": "linestyle",
        },
        {
            "input": (
                {"mfc": "blue", "c": "blue", "line_width": 2},
                "plot",
            ),
            "output": ("markerfacecolor", "color", "line_width"),
        },
        {"input": ({"ec": "blue", "fc": "black"}, "hist"), "output": ("edgecolor", "facecolor")},
        {
            "input": ({"edgecolors": "blue", "lw": 3}, "hlines"),
            "output": ("edgecolor", "linewidth"),
        },
    ],
)
def test_matplotlib_kwarg_dealiaser(params):
    dealiased = matplotlib_kwarg_dealiaser(params["input"][0], kind=params["input"][1])
    for returned in dealiased:
        assert returned in params["output"]


@pytest.mark.parametrize("c_values", ["#0000ff", "blue", [0, 0, 1]])
def test_vectorized_to_hex_scalar(c_values):
    output = vectorized_to_hex(c_values)
    assert output == "#0000ff"


@pytest.mark.parametrize(
    "c_values", [["blue", "blue"], ["blue", "#0000ff"], np.array([[0, 0, 1], [0, 0, 1]])]
)
def test_vectorized_to_hex_array(c_values):
    output = vectorized_to_hex(c_values)
    assert np.all([item == "#0000ff" for item in output])


def test_mpl_dealiase_sel_kwargs():
    """Check mpl dealiase_sel_kwargs behaviour.

    Makes sure kwargs are overwritten when necessary even with alias involved and that
    they are not modified when not included in props.
    """
    kwargs = {"linewidth": 3, "alpha": 0.4, "line_color": "red"}
    props = {"lw": [1, 2, 4, 5], "linestyle": ["-", "--", ":"]}
    res = dealiase_sel_kwargs(kwargs, props, 2)
    assert "linewidth" in res
    assert res["linewidth"] == 4
    assert "linestyle" in res
    assert res["linestyle"] == ":"
    assert "alpha" in res
    assert res["alpha"] == 0.4
    assert "line_color" in res
    assert res["line_color"] == "red"


@pytest.mark.skipif(
    not (bokeh_installed or running_on_ci()),
    reason="test requires bokeh which is not installed",
)
def test_bokeh_dealiase_sel_kwargs():
    """Check bokeh dealiase_sel_kwargs behaviour.

    Makes sure kwargs are overwritten when necessary even with alias involved and that
    they are not modified when not included in props.
    """
    from ...plots.backends.bokeh import dealiase_sel_kwargs

    kwargs = {"line_width": 3, "line_alpha": 0.4, "line_color": "red"}
    props = {"line_width": [1, 2, 4, 5], "line_dash": ["dashed", "dashed", "dashed"]}
    res = dealiase_sel_kwargs(kwargs, props, 2)
    assert "line_width" in res
    assert res["line_width"] == 4
    assert "line_dash" in res
    assert res["line_dash"] == "dashed"
    assert "line_alpha" in res
    assert res["line_alpha"] == 0.4
    assert "line_color" in res
    assert res["line_color"] == "red"


@pytest.mark.skipif(
    not (bokeh_installed or running_on_ci()),
    reason="test requires bokeh which is not installed",
)
def test_set_bokeh_circular_ticks_labels():
    """Assert the axes returned after placing ticks and tick labels for circular plots."""
    import bokeh.plotting as bkp

    ax = bkp.figure(x_axis_type=None, y_axis_type=None)
    hist = np.linspace(0, 1, 10)
    labels = ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
    ax = set_bokeh_circular_ticks_labels(ax, hist, labels)
    renderers = ax.renderers
    assert len(renderers) == 3
    assert renderers[2].data_source.data["text"] == labels
    assert len(renderers[0].data_source.data["start_angle"]) == len(labels)


def test_compute_ranks():
    pois_data = np.array([[5, 4, 1, 4, 0], [2, 8, 2, 1, 1]])
    expected = np.array([[9.0, 7.0, 3.0, 8.0, 1.0], [5.0, 10.0, 6.0, 2.0, 4.0]])
    ranks = compute_ranks(pois_data)
    np.testing.assert_equal(ranks, expected)

    norm_data = np.array(
        [
            [0.2644187, -1.3004813, -0.80428456, 1.01319068, 0.62631143],
            [1.34498018, -0.13428933, -0.69855487, -0.9498981, -0.34074092],
        ]
    )
    expected = np.array([[7.0, 1.0, 3.0, 9.0, 8.0], [10.0, 6.0, 4.0, 2.0, 5.0]])
    ranks = compute_ranks(norm_data)
    np.testing.assert_equal(ranks, expected)
