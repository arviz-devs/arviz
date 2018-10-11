"""Configuration for test suite."""
# pylint: disable=redefined-outer-name
import os
import pytest


def pytest_addoption(parser):
    """Definition for command line option to save figures from tests."""
    parser.addoption("--save", nargs="?", const="test_images", help="Save images rendered by plot")


@pytest.fixture(scope="session")
def save_figs(request):
    """Enable command line switch for saving generation figures upon testing."""
    fig_dir = request.config.getoption("--save")

    if fig_dir is not None:
        # Try creating directory if it doesn't exist
        print("Saving generated images in {}".format(fig_dir))

        os.makedirs(fig_dir, exist_ok=True)
        print("Directory {} created".format(fig_dir))

        # Clear all files from the directory
        for file in os.listdir(fig_dir):
            full_path = os.path.join(fig_dir, file)

            try:
                os.remove(full_path)

            except OSError:
                print("Failed to remove {}".format(full_path))

    return fig_dir
