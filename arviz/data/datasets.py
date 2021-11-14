"""Base IO code for all datasets. Heavily influenced by scikit-learn's implementation."""
import hashlib
import itertools
import os
import shutil
from collections import namedtuple
from urllib.request import urlretrieve

from ..rcparams import rcParams
from .io_netcdf import from_netcdf

LocalFileMetadata = namedtuple("LocalFileMetadata", ["filename", "description"])

RemoteFileMetadata = namedtuple(
    "RemoteFileMetadata", ["filename", "url", "checksum", "description"]
)
_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_datasets")

# Models for local datasets are located at https://github.com/arviz-devs/arviz_example_data

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
        filename="radon_hierarchical.nc",
        url="http://ndownloader.figshare.com/files/24067472",
        checksum="a9b2b4adf1bf9c5728e5bdc97107e69c4fc8d8b7d213e9147233b57be8b4587b",
        description="""
Radon is a radioactive gas that enters homes through contact points with the ground.
It is a carcinogen that is the primary cause of lung cancer in non-smokers. Radon
levels vary greatly from household to household.

This example uses an EPA study of radon levels in houses in Minnesota to construct a
model with a hierarchy over households within a county. The model includes estimates
(gamma) for contextual effects of the uranium per household.

See Gelman and Hill (2006) for details on the example, or
https://docs.pymc.io/notebooks/multilevel_modeling.html
by Chris Fonnesbeck for details on this implementation.
""",
    ),
    "rugby": RemoteFileMetadata(
        filename="rugby.nc",
        url="http://ndownloader.figshare.com/files/16254359",
        checksum="9eecd2c6317e45b0388dd97ae6326adecf94128b5a7d15a52c9fcfac0937e2a6",
        description="""
The Six Nations Championship is a yearly rugby competition between Italy, Ireland,
Scotland, England, France and Wales. Fifteen games are played each year, representing
all combinations of the six teams.

This example uses and includes results from 2014 - 2017, comprising 60 total
games. It models latent parameters for each team's attack and defense, as well
as a parameter for home team advantage.

See https://docs.pymc.io/notebooks/rugby_analytics.html by Peader Coyle
for more details and references.
""",
    ),
    "regression1d": RemoteFileMetadata(
        filename="regression1d.nc",
        url="http://ndownloader.figshare.com/files/16254899",
        checksum="909e8ffe344e196dad2730b1542881ab5729cb0977dd20ba645a532ffa427278",
        description="""
A synthetic one dimensional linear regression dataset with latent slope,
intercept, and noise ("eps"). One hundred data points, fit with PyMC3.

True slope and intercept are included as deterministic variables.
""",
    ),
    "regression10d": RemoteFileMetadata(
        filename="regression10d.nc",
        url="http://ndownloader.figshare.com/files/16255736",
        checksum="c6716ec7e19926ad2a52d6ae4c1d1dd5ddb747e204c0d811757c8e93fcf9f970",
        description="""
A synthetic multi-dimensional (10 dimensions) linear regression dataset with
latent weights ("w"), intercept, and noise ("eps"). Five hundred data points,
fit with PyMC3.

True weights and intercept are included as deterministic variables.
""",
    ),
    "classification1d": RemoteFileMetadata(
        filename="classification1d.nc",
        url="http://ndownloader.figshare.com/files/16256678",
        checksum="1cf3806e72c14001f6864bb69d89747dcc09dd55bcbca50aba04e9939daee5a0",
        description="""
A synthetic one dimensional logistic regression dataset with latent slope and
intercept, passed into a Bernoulli random variable. One hundred data points,
fit with PyMC3.

True slope and intercept are included as deterministic variables.
""",
    ),
    "classification10d": RemoteFileMetadata(
        filename="classification10d.nc",
        url="http://ndownloader.figshare.com/files/16256681",
        checksum="16c9a45e1e6e0519d573cafc4d266d761ba347e62b6f6a79030aaa8e2fde1367",
        description="""
A synthetic multi dimensional (10 dimensions) logistic regression dataset with
latent weights ("w") and intercept, passed into a Bernoulli random variable.
Five hundred data points, fit with PyMC3.

True weights and intercept are included as deterministic variables.
""",
    ),
    "glycan_torsion_angles": RemoteFileMetadata(
        filename="glycan_torsion_angles.nc",
        url="http://ndownloader.figshare.com/files/22882652",
        checksum="4622621fe7a1d3075c18c4c34af8cc57c59eabbb3501b20c6e2d9c6c4737034c",
        description="""
Torsion angles phi and psi are critical for determining the three dimensional
structure of bio-molecules. Combinations of phi and psi torsion angles that
produce clashes between atoms in the bio-molecule result in high energy, unlikely structures.

This model uses a Von Mises distribution to propose torsion angles for the
structure of a glycan molecule (pdb id: 2LIQ), and a Potential to estimate
the proposed structure's energy. Said Potential is bound by Boltzman's law.
""",
    ),
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


def load_arviz_data(dataset=None, data_home=None, regex=False, **kwargs):
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

    regex : bool, optional
        Specifies regex support for chunking information in
        :func:`arviz.from_netcdf`. This feature is currently experimental.

    **kwargs : dict of {str: dict}, optional
        Keyword arguments to be passed to :func:`arviz.from_netcdf`.
        This feature is currently experimental.

    Returns
    -------
    xarray.Dataset

    """
    if dataset in LOCAL_DATASETS:
        resource = LOCAL_DATASETS[dataset]
        return from_netcdf(resource.filename)

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
                "{} has an SHA256 checksum ({}) differing from expected ({}), "
                "file may be corrupted. Run `arviz.clear_data_home()` and try "
                "again, or please open an issue.".format(file_path, checksum, remote.checksum)
            )
        return from_netcdf(file_path, kwargs, regex)
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
        lines.append(f"{name}\n{'=' * len(name)}\n{resource.description}\n{location}")

    return f"\n\n{10 * '-'}\n\n".join(lines)
