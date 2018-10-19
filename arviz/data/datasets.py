"""Base IO code for all datasets. Heavily influenced by scikit-learn's implementation."""
from collections import namedtuple
import hashlib
import itertools
import os
import shutil
from urllib.request import urlretrieve

from .io_netcdf import load_data

LocalFileMetadata = namedtuple("LocalFileMetadata", ["filename", "description"])

RemoteFileMetadata = namedtuple(
    "RemoteFileMetadata", ["filename", "url", "checksum", "description"]
)
_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_datasets")

LOCAL_DATASETS = {
    "centered_eight": LocalFileMetadata(
        filename=os.path.join(_DATASET_DIR, "centered_eight.nc"),
        description="""
A centered parameterization of the eight schools model. Provided as an example of a
model that NUTS has trouble fitting. Compare to `load_arviz_data("non_centered_eight")`.

The eight schools model is a hierarchical model used for an analysis of the effectiveness
of classes that were designed to improve students’ performance on the Scholastic Aptitude Test.

See Bayesian Data Analysis (Gelman et. al.) for more details.
""",
    ),
    "non_centered_eight": LocalFileMetadata(
        filename=os.path.join(_DATASET_DIR, "non_centered_eight.nc"),
        description="""
A non-centered parameterization of the eight schools model. This is a hierarchical model
where sampling problems may be fixed by a non-centered parametrization. Compare to
`load_arviz_data("centered_eight")`.

The eight schools model is a hierarchical model used for an analysis of the effectiveness
of classes that were designed to improve students’ performance on the Scholastic Aptitude Test.

See Bayesian Data Analysis (Gelman et. al.) for more details.
""",
    ),
}

REMOTE_DATASETS = {
    "radon": RemoteFileMetadata(
        filename="radon.nc",
        url="https://ndownloader.figshare.com/files/13284311",
        checksum="ee9d4644e498d45ab5163982fc74baf05efce5cfa87b11f8509f7b9acf471f09",
        description="""
Radon is a radioactive gas that enters homes through contact points with the ground.
It is a carcinogen that is the primary cause of lung cancer in non-smokers. Radon
levels vary greatly from household to household.

This example uses an EPA study of radon levels in houses in Minnesota to construct a
model with a hierarchy over households within a county. The model includes estimates
(gamma) for contextual effects of the uranium per household.

See Gelman and Hill (2006) for details on the example, or
https://docs.pymc.io/notebooks/multilevel_modeling.html#Correlations-among-levels
by Chris Fonnesbeck for details on this implementation.
""",
    )
}


def get_data_home(data_home=None):
    """Return the path of the arviz data dir.

    This folder is used by some dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'arviz_data' in the
    user home folder.

    Alternatively, it can be set by the 'ARVIZ_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str | None
        The path to arviz data dir.
    """
    if data_home is None:
        data_home = os.environ.get("ARVIZ_DATA", os.path.join("~", "arviz_data"))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str | None
        The path to arviz data dir.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as buff:
        while True:
            buffer = buff.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def load_arviz_data(dataset=None, data_home=None):
    """Load a local or remote pre-made dataset.

    Run with no parameters to get a list of all available models.

    The directory to save to can also be set with the environement
    variable `ARVIZ_HOME`. The checksum of the dataset is checked against a
    hardcoded value to watch for data corruption.

    Run `az.clear_data_home` to clear the data directory.

    Parameters
    ----------
    dataset : str
        Name of dataset to load.

    data_home : str, optional
        Where to save remote datasets

    Returns
    -------
    xarray.Dataset
    """
    if dataset in LOCAL_DATASETS:
        resource = LOCAL_DATASETS[dataset]
        return load_data(resource.filename)

    elif dataset in REMOTE_DATASETS:
        remote = REMOTE_DATASETS[dataset]
        home_dir = get_data_home(data_home=data_home)
        file_path = os.path.join(home_dir, remote.filename)
        if not os.path.exists(file_path):
            urlretrieve(remote.url, file_path)
        checksum = _sha256(file_path)
        if remote.checksum != checksum:
            raise IOError(
                "{} has an SHA256 checksum ({}) differing from expected ({}), "
                "file may be corrupted. Run `arviz.clear_data_home()` and try "
                "again, or please open an issue.".format(file_path, checksum, remote.checksum)
            )
        return load_data(file_path)
    else:
        raise ValueError(
            "Dataset {} not found! The following are available:\n{}".format(
                dataset, list_datasets()
            )
        )


def list_datasets():
    """Get a string representation of all available datasets with descriptions."""
    lines = []
    for name, resource in itertools.chain(LOCAL_DATASETS.items(), REMOTE_DATASETS.items()):

        if isinstance(resource, LocalFileMetadata):
            location = "local: {}".format(resource.filename)
        elif isinstance(resource, RemoteFileMetadata):
            location = "remote: {}".format(resource.url)
        else:
            location = "unknown"
        lines.append("{}\n{}\n{}\n{}".format(name, "=" * len(name), resource.description, location))

    return "\n\n{}\n\n".format(10 * "-").join(lines)
