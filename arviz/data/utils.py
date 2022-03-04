"""Data specific utilities."""
import numpy as np

from ..utils import _var_names
from .converters import convert_to_dataset


def extract_dataset(
    data,
    group="posterior",
    combined=True,
    var_names=None,
    filter_vars=None,
    num_samples=None,
    rng=None,
):
    """Extract an InferenceData group or subset of it as a :class:`xarray.Dataset`.

    Parameters
    ----------
    idata : InferenceData or InferenceData_like
        InferenceData from which to extract the data.
    group : str, optional
        Which InferenceData data group to extract data from.
    combined : bool, optional
        Combine ``chain`` and ``draw`` dimensions into ``sample``. Won't work if
        a dimension named ``sample`` already exists.
    var_names : str or list of str, optional
        Variables to be plotted, two variables are required. Prefix the variables by `~`
        when you want to exclude them from the plot.
    filter_vars: {None, "like", "regex"}, optional
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
        Like with plotting, sometimes it's easier to subset saying what to exclude
        instead of what to include
    num_samples : int, optional
        Extract only a subset of the samples. Only valid if ``combined=True``
    rng : bool, int, numpy.Generator, optional
        Shuffle the samples, only valid if ``combined=True``. By default,
        samples are shuffled if ``num_samples`` is not ``None``, and are left
        in the same order otherwise. This ensures that subsetting the samples doesn't return
        only samples from a single chain and consecutive draws.

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    The default behaviour is to return the posterior group after stacking the chain and
    draw dimensions.

    .. jupyter-execute::

        import arviz as az
        idata = az.load_arviz_data("centered_eight")
        az.extract_dataset(idata)

    You can also indicate a subset to be returned, but in variables and in samples:

    .. jupyter-execute::

        az.extract_dataset(idata, var_names="theta", num_samples=100)

    To keep the chain and draw dimensions, use ``combined=False``.

    .. jupyter-execute::

        az.extract_dataset(idata, group="prior", combined=False)

    """
    if num_samples is not None and not combined:
        raise ValueError("num_samples is only compatible with combined=True")
    if rng is None:
        rng = num_samples is not None
    if rng is not False and not combined:
        raise ValueError("rng is only compatible with combined=True")
    data = convert_to_dataset(data, group=group)
    var_names = _var_names(var_names, data, filter_vars)
    if var_names is not None:
        data = data[var_names]
    if combined:
        data = data.stack(sample=("chain", "draw"))
    # 0 is a valid seed se we need to check for rng being exactly boolean
    if rng is not False:
        if rng is True:
            rng = np.random.default_rng()
        # default_rng takes ints or sequences of ints
        try:
            rng = np.random.default_rng(rng)
            random_subset = rng.permutation(np.arange(len(data["sample"])))
        except TypeError as err:
            raise TypeError("Unable to initializate numpy random Generator from rng") from err
        except AttributeError as err:
            raise AttributeError("Unable to use rng to generate a permutation") from err
        data = data.isel(sample=random_subset)
    if num_samples is not None:
        data = data.isel(sample=slice(None, num_samples))
    return data
