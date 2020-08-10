# pylint: disable=redefined-outer-name
"""Configuration for test suite."""
import logging
import os

import numpy as np
import pytest

_log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def random_seed():
    """Reset numpy random seed generator."""
    np.random.seed(0)


def pytest_addoption(parser):
    """Definition for command line option to save figures from tests."""
    parser.addoption("--save", nargs="?", const="test_images", help="Save images rendered by plot")


@pytest.fixture(scope="session")
def save_figs(request):
    """Enable command line switch for saving generation figures upon testing."""
    fig_dir = request.config.getoption("--save")

    if fig_dir is not None:
        # Try creating directory if it doesn't exist
        _log.info("Saving generated images in %s", fig_dir)

        os.makedirs(fig_dir, exist_ok=True)
        _log.info("Directory %s created", fig_dir)

        # Clear all files from the directory
        # Does not alter or delete directories
        for file in os.listdir(fig_dir):
            full_path = os.path.join(fig_dir, file)

            try:
                os.remove(full_path)

            except OSError:
                _log.info("Failed to remove %s", full_path)

    return fig_dir
