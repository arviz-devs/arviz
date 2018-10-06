"""
Configuration for test suite
"""
import pytest
import os


def pytest_addoption(parser):
    """Definition for command line option to save figures from tests"""
    parser.addoption("--save", nargs="?", const="test_images",
                     help="Save images rendered by plot")


@pytest.fixture(scope="session")
def save_figs(request):
    """Enables command line switch for saving generation figures upon testing"""
    fig_dir = request.config.getoption("--save")

    if fig_dir is not None:
        # Try creating directory if it doesn't exist
        try:
            os.mkdir(fig_dir)
            print("Directory {} created".format(fig_dir))

        except FileExistsError:
            print("Directory {} already exists".format(fig_dir))

        # Clear all files from the directory
        for file in os.listdir(fig_dir):
            try:
                os.remove(file)
            except OSError:
                print("Can't remove {}".format(file))

    else:
        print("Not saving figures")

    return fig_dir


