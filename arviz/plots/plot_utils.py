import numpy as np
import matplotlib.pyplot as plt


def get_axis(ax, default_rows, default_columns, **default_kwargs):
    """Verifies the provided axis is of the correct shape, and creates one if needed.

    Args:
        ax: matplotlib axis or None
        default_rows: int, expected rows in axis
        default_columns: int, expected columns in axis
        **default_kwargs: keyword arguments to pass to plt.subplot

    Returns:
        axis, or raises an error
    """

    default_shape = (default_rows, default_columns)
    if ax is None:
        _, ax = plt.subplots(*default_shape, **default_kwargs)
    elif ax.shape != default_shape:
        raise ValueError('Subplots with shape %r required' % (default_shape,))
    return ax


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
    if  x_range > max_bins:
        bins = range(x_min, x_max + fenceposts, int(x_range / 10))
    else:
        bins = range(x_min, x_max + fenceposts)
    return bins
