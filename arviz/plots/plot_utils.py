"""Utilities for plotting."""
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def make_2d(ary):
    """Convert any array into a 2d numpy array.

    In case the array is already more than 2 dimensional, will ravel the
    dimensions after the first.
    """
    dim_0, *_ = np.atleast_1d(ary).shape
    return ary.reshape(dim_0, -1, order='F')


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
    """Compute number of bins (or ticks).

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
        bins = range(x_min, x_max + fenceposts, max(1, int(x_range / 10)))
    else:
        bins = range(x_min, x_max + fenceposts)
    return bins


def default_grid(n_items, max_cols=6, min_cols=3):
    """Make a grid for subplots.

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
        return np.clip(val, min_cols, max_cols)

    if n_items <= max_cols:
        return 1, n_items
    ideal = in_bounds(round(n_items ** 0.5))

    for offset in (0, 1, -1, 2, -2):
        cols = in_bounds(ideal + offset)
        rows, extra = divmod(n_items, cols)
        if extra == 0:
            return rows, cols
    return n_items // ideal + 1, ideal


def _create_axes_grid(length_plotters, rows, cols, **kwargs):
    """Create figure and axes for grids with multiple plots.

    Parameters
    ----------
    n_items : int
        Number of panels required
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axes
    """
    fig, ax = plt.subplots(rows, cols, **kwargs)
    ax = np.ravel(ax)
    extra = (rows * cols) - length_plotters
    if extra:
        for i in range(1, extra+1):
            ax[-i].set_axis_off()
        ax = ax[:-extra]
    return fig, ax


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
    return ', '.join(['{}'.format(v) for _, v in selection.items()])


def make_label(var_name, selection):
    """Consistent labelling for plots.

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
        return '{}\n({})'.format(var_name, selection_to_string(selection))
    return '{}'.format(var_name)


def xarray_var_iter(data, var_names=None, combined=False, skip_dims=None, reverse_selections=False):
    """Convert xarray data to an iterator over vectors.

    Iterates over each var_name and all of its coordinates, returning the 1d
    data.

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

    reverse_selections : bool
        Whether to reverse selections before iterating.

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
            new_dims = [dim for dim in data[var_name].dims if dim not in skip_dims]
            vals = [data[var_name][dim].values for dim in new_dims]
            dims = [{k : v for k, v in zip(new_dims, prod)} for prod in product(*vals)]
            if reverse_selections:
                dims = reversed(dims)

            for selection in dims:
                yield var_name, selection, data[var_name].sel(**selection).values


def xarray_to_ndarray(data, *, var_names=None, combined=True):
    """Take xarray data and unpacks into variables and data into list and numpy array respectively.

    Assumes that chain and draw are in coordinates

    Parameters
    ----------
    data: xarray.DataSet
        Data in an xarray from an InferenceData object. Examples include posterior or sample_stats

    var_names: iter
        Should be a subset of data.data_vars not including chain and draws. Defaults to all of them

    combined: bool
        Whether to combine chain into one array

    Returns
    -------
    var_names: list
        List of variable names
    data: np.array
        Data values
    """
    unpacked_data, unpacked_var_names, = [], []

    # Merge chains and variables
    for var_name, selection, data_array in xarray_var_iter(data, var_names=var_names,
                                                           combined=combined):
        unpacked_data.append(data_array.flatten())
        unpacked_var_names.append(make_label(var_name, selection))

    return unpacked_var_names, np.array(unpacked_data)


def get_coords(data, coords):
    """Subselects xarray dataset object to provided coords. Raises exception if fails.

    Raises
    ------
    ValueError
        If coords name are not available in data

    KeyError
        If coords dims are not available in data

    Returns
    -------
    data: xarray
        xarray.Dataset object
    """
    try:
        return data.sel(**coords)

    except ValueError:
        invalid_coords = set(coords.keys()) - set(data.coords.keys())
        raise ValueError("Coords {} are invalid coordinate keys".format(invalid_coords))

    except KeyError as err:
        raise KeyError(("Coords should follow mapping format {{coord_name:[dim1, dim2]}}. "
                        "Check that coords structure is correct and"
                        " dimensions are valid. {}").format(err))
