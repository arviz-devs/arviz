# pylint: disable=redefined-outer-name
import numpy as np
import xarray as xr
import pytest

from ..plots.plot_utils import xarray_to_nparray


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
