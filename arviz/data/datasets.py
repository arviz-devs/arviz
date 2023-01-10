"""Base IO code for all datasets. Heavily influenced by scikit-learn's implementation."""
import hashlib
import itertools
import json
import os
import shutil
from collections import namedtuple
from urllib.request import urlretrieve

from ..rcparams import rcParams
from .io_netcdf import from_netcdf

LocalFileMetadata = namedtuple("LocalFileMetadata", ["name", "filename", "description"])

RemoteFileMetadata = namedtuple(
    "RemoteFileMetadata", ["name", "filename", "url", "checksum", "description"]
)
_EXAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_data")
_LOCAL_DATA_DIR = os.path.join(_EXAMPLE_DATA_DIR, "data")

with open(os.path.join(_EXAMPLE_DATA_DIR, "data_local.json"), "r", encoding="utf-8") as f:
    LOCAL_DATASETS = {
        entry["name"]: LocalFileMetadata(
            name=entry["name"],
            filename=os.path.join(_LOCAL_DATA_DIR, entry["filename"]),
            description=entry["description"],
        )
        for entry in json.load(f)
    }

with open(os.path.join(_EXAMPLE_DATA_DIR, "data_remote.json"), "r", encoding="utf-8") as f:
    REMOTE_DATASETS = {entry["name"]: RemoteFileMetadata(**entry) for entry in json.load(f)}


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


def load_arviz_data(dataset=None, data_home=None, **kwargs):
    """Load a local or remote pre-made dataset.

    Run with no parameters to get a list of all available models.

    The directory to save to can also be set with the environment
    variable `ARVIZ_HOME`. The checksum of the dataset is checked against a
    hardcoded value to watch for data corruption.

    Run `az.clear_data_home` to clear the data directory.

    Parameters
    ----------
    dataset : str
        Name of dataset to load.
    data_home : str, optional
        Where to save remote datasets
    **kwargs : dict, optional
        Keyword arguments passed to :func:`arviz.from_netcdf`.

    Returns
    -------
    xarray.Dataset

    """
    if dataset in LOCAL_DATASETS:
        resource = LOCAL_DATASETS[dataset]
        return from_netcdf(resource.filename, **kwargs)

    elif dataset in REMOTE_DATASETS:
        remote = REMOTE_DATASETS[dataset]
        home_dir = get_data_home(data_home=data_home)
        file_path = os.path.join(home_dir, remote.filename)

        if not os.path.exists(file_path):
            http_type = rcParams["data.http_protocol"]

            # Replaces http type. Redundant if http_type is http, useful if http_type is https
            url = remote.url.replace("http", http_type)
            urlretrieve(url, file_path)

        checksum = _sha256(file_path)
        if remote.checksum != checksum:
            raise IOError(
                f"{file_path} has an SHA256 checksum ({checksum}) differing from expected "
                "({remote.checksum}), file may be corrupted. "
                "Run `arviz.clear_data_home()` and try again, or please open an issue."
            )
        return from_netcdf(file_path, **kwargs)
    else:
        if dataset is None:
            return dict(itertools.chain(LOCAL_DATASETS.items(), REMOTE_DATASETS.items()))
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
            location = f"local: {resource.filename}"
        elif isinstance(resource, RemoteFileMetadata):
            location = f"remote: {resource.url}"
        else:
            location = "unknown"
        lines.append(f"{name}\n{'=' * len(name)}\n{resource.description}\n\n{location}")

    return f"\n\n{10 * '-'}\n\n".join(lines)
