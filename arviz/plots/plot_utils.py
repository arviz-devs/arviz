import itertools

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def make_2d(ary):
    """Convert any array into a 2d numpy array.

    In case the array is already more than 2 dimensional, will ravel the
    dimensions after the first.
    """
    ary = np.atleast_2d(ary.T).T
    # flatten out dimensions beyond the first
    first_dim = ary.shape[0]
    newshape = np.product(ary.shape[1:]).astype(int)
    ary = ary.reshape((first_dim, newshape), order='F')
    return ary


def _scale_text(figsize, textsize, scale_ratio=2):
    """Scale text and linewidth to figsize.

    Parameters
    ----------
    figsize : float or None
        Size of figure in inches
    textsize : float or None
        Desired text size
    scale_ratio : float (default: 2)
        Ratio of size of elements compared to figsize.  Larger is bigger.
    """

    if textsize is None and figsize is not None:
        textsize = figsize[0] * scale_ratio

    linewidth = textsize / 8
    markersize = textsize / 2
    return textsize, linewidth, markersize


def get_bins(ary, max_bins=50, fenceposts=2):
    """
    Compute number of bins (or ticks)

    Parameters
    ----------
    ary : numpy.array
        array to be binned
    max_bins : int
        maximum number of bins
    fenceposts : int
        when computing bins, this should be 2, when computing ticks this should be 1.
    """
    x_max, x_min = ary.max(), ary.min()
    x_range = x_max - x_min
    if x_range > max_bins:
        bins = range(x_min, x_max + fenceposts, int(x_range / 10))
    else:
        bins = range(x_min, x_max + fenceposts)
    return bins


def _create_axes_grid(trace, figsize, ax):
    """
    Parameters
    ----------
    trace : dict or DataFrame
        dictionary with ppc samples of DataFrame with posterior samples
    figsize : tuple
        figure size
    ax : matplotlib axes

    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axes
    """
    if isinstance(trace, dict):
        l_trace = len(trace)
    else:
        l_trace = trace.shape[1]
    if figsize is None:
        figsize = (8, 2 + l_trace + (l_trace % 2))
    if ax is None:
        if l_trace == 1:
            _, ax = plt.subplots(figsize=figsize)
        else:
            n_rows = np.ceil(l_trace / 2.0).astype(int)
            _, ax = plt.subplots(n_rows, 2, figsize=figsize)
            ax = ax.reshape(2 * n_rows)
            if l_trace % 2 == 1:
                ax[-1].set_axis_off()
                ax = ax[:-1]
    return ax, figsize


def default_grid(n_items, max_cols=6, min_cols=3):
    """Makes a grid for subplots

    Tries to get as close to sqrt(n_items) x sqrt(n_items) as it can,
    but allows for custom logic

    Parameters
    ----------
    n_items : int
        Number of panels required
    max_cols : int
        Maximum number of columns, inclusive
    min_cols : int
        Minimum number of columns, inclusive

    Returns
    -------
    (int, int)
        Rows and columns, so that rows * columns >= n_items
    """
    def in_bounds(val):
        return max(min(val, max_cols), min_cols)

    if n_items <= max_cols:
        return 1, n_items
    ideal = in_bounds(int(np.round(n_items ** 0.5)))

    for offset in (0, 1, -1, 2, -2):
        cols = in_bounds(ideal + offset)
        rows, extra = divmod(n_items, cols)
        if extra == 0:
            return rows, cols
    return n_items // ideal + 1, ideal


def selection_to_string(selection):
    """Convert dictionary of coordinates to a string for labels.

    Parameters
    ----------
    selection : dict[Any] -> Any

    Returns
    -------
    str
        key1: value1, key2: value2, ...
    """
    return ', '.join(['{}: {}'.format(k, v) for k, v in selection.items()])


def make_label(var_name, selection):
    """Consistent labelling for plots

    Parameters
    ----------
    var_name : str
       Name of the variable

    selection : dict[Any] -> Any
        Coordinates of the variable

    Returns
    -------
    str
        A text representation of the label
    """
    if selection:
        return f'{var_name} ({selection_to_string(selection)})'
    return f'{var_name}'


def xarray_var_iter(data, var_names=None, combined=False, skip_dims=None):
    """Converts xarray data to an iterator over vectors

    Iterates over each var_name and all of its coordinates, returning the 1d
    data

    Parameters
    ----------
    data : xarray.Dataset
        Posterior data in an xarray

    var_names : iterator of strings (optional)
        Should be a subset of data.data_vars. Defaults to all of them.

    combined : bool
        Whether to combine chains or leave them separate

    skip_dims : set
        dimensions to not iterate over

    Returns
    -------
    Iterator of (str, dict(str, any), np.array)
        The string is the variable name, the dictionary are coordinate names to values,
        and the array are the values of the variable at those coordinates.
    """
    if skip_dims is None:
        skip_dims = set()

    if combined:
        skip_dims = skip_dims.union({'chain', 'draw'})
    else:
        skip_dims.add('draw')

    if var_names is None:
        if isinstance(data, xr.Dataset):
            var_names = list(data.data_vars)
        elif isinstance(data, xr.DataArray):
            var_names = [data.name]
            data = {data.name: data}

    for var_name in var_names:
        if var_name in data:
            new_dims = set(data[var_name].dims) - skip_dims
            vals = [data[var_name][dim].values for dim in new_dims]
            dims = [{k: v for k, v in zip(new_dims, prod)} for prod in itertools.product(*vals)]
            for selection in dims:
                yield var_name, selection, data[var_name].sel(**selection).values
