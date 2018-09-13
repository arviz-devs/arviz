# pylint: disable=redefined-outer-name
import numpy as np
import xarray as xr
import pytest

from ..plots.plot_utils import xarray_to_nparray, xarray_var_iter, get_coords


@pytest.fixture(scope='function')
def sample_dataset():
    mu = np.arange(1, 7).reshape(2, 3)
    tau = np.arange(7, 13).reshape(2, 3)

    chain = [0, 1]
    draws = [0, 1, 2]

    data = xr.Dataset({"mu": (["chain", "draw"], mu), "tau": (["chain", "draw"], tau)},
                      coords={"draw": draws, "chain": chain})

    return mu, tau, data


def test_dataset_to_numpy_not_combined(sample_dataset):  # pylint: disable=invalid-name
    mu, tau, data = sample_dataset
    var_names, data = xarray_to_nparray(data, combined=False)

    # 2 vars x 2 chains
    assert len(var_names) == 4
    assert (data == np.concatenate((mu, tau), axis=0)).all()


def test_dataset_to_numpy_combined(sample_dataset):
    mu, tau, data = sample_dataset
    var_names, data = xarray_to_nparray(data, combined=True)

    assert len(var_names) == 2
    assert (data[0] == mu.reshape(1, 6)).all()
    assert (data[1] == tau.reshape(1, 6)).all()


def test_xarray_var_iter_ordering_combined(sample_dataset):  # pylint: disable=invalid-name
    """Assert that varname order stays consistent when chains are combined"""
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_var_iter(data, var_names=None, combined=True)]
    assert var_names == ["mu", "tau"]


def test_xarray_var_iter_ordering_uncombined(sample_dataset):  # pylint: disable=invalid-name
    """Assert that varname order stays consistent when chains are not combined"""
    _, _, data = sample_dataset
    var_names = [(var, selection) for (var, selection, _) in xarray_var_iter(data, var_names=None)]
    assert var_names == [("mu", {"chain": 0}), ("mu", {"chain": 1}),
                         ("tau", {"chain": 0}), ("tau", {"chain": 1})]


class TestCoordsExceptions:
    def test_invalid_coord_name(self, sample_dataset):  # pylint: disable=invalid-name
        """Assert that nicer exception appears when user enters wrong coords name"""
        _, _, data = sample_dataset
        coords = {"NOT_A_COORD_NAME": [1]}

        with pytest.raises(ValueError) as err:
            get_coords(data, coords)

        assert "Verify coords keys" in str(err)

    def test_invalid_coord_value(self, sample_dataset):  # pylint: disable=invalid-name
        """Assert that nicer exception appears when user enters wrong coords value"""
        _, _, data = sample_dataset
        coords = {"draw": [1234567]}

        with pytest.raises(KeyError) as err:
            get_coords(data, coords)

        assert "Verify coords values" in str(err)
