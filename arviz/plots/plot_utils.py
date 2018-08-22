import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ..utils import selection_to_string


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
        bins = range(x_min, x_max + fenceposts, max(1, int(x_range / 10)))
    else:
        bins = range(x_min, x_max + fenceposts)
    return bins


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


def _create_axes_grid(length_plotters, rows, cols, **kwargs):
    """
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
        return '{} ({})'.format(var_name, selection_to_string(selection))
    return '{}'.format(var_name)
