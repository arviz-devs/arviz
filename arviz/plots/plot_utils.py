import numpy as np
import matplotlib.pyplot as plt


def identity_transform(x):
    """f(x) = x"""
    return x


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


def make_2d(a):
    """Ravel the dimensions after the first."""
    a = np.atleast_2d(a.T).T
    # flatten out dimensions beyond the first
    n = a.shape[0]
    newshape = np.product(a.shape[1:]).astype(int)
    a = a.reshape((n, newshape), order='F')
    return a


def _scale_text(figsize, text_size):
    """Scale text to figsize."""

    if text_size is None and figsize is not None:
        if figsize[0] <= 11:
            return 12
        else:
            return figsize[0]
    else:
        return text_size

def get_bins(x, max_bins=50, n=2):
    """
    Compute number of bins (or ticks)

    Parameters
    ----------
    x : array
        array to be binned
    max_bins : int
        maximum number of bins
    n : int
        when computing bins, this should be 2, when computing ticks this should be 1.
    """
    x_max, x_min = x.max(), x.min()
    x_range = x_max - x_min
    if  x_range > cutoff:
        bins = range(x_min, x_max + n, int(x_range / 10))
    else:
        bins = range(x_min, x_max + n)
    return bins
