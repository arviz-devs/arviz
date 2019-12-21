"""Matplotlib Autocorrplot."""
import warnings
import numpy as np

from ....stats import autocorr
from ...plot_utils import make_label


def plot_autocorr(
    axes,
    plotters,
    max_lag,
    linewidth,
    titlesize,
    combined=False,
    xt_labelsize=None,
    backend_kwargs=None,
):
    """Matplotlib autocorrplot."""
    if backend_kwargs is not None:
        warnings.warn(
            (
                "Argument backend_kwargs has not effect in matplotlib.plot_autocorr"
                "Supplied value won't be used"
            )
        )
    for (var_name, selection, x), ax_ in zip(plotters, axes.flatten()):
        x_prime = x
        if combined:
            x_prime = x.flatten()
        y = autocorr(x_prime)
        ax_.vlines(x=np.arange(0, max_lag), ymin=0, ymax=y[0:max_lag], lw=linewidth)
        ax_.hlines(0, 0, max_lag, "steelblue")
        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax_.tick_params(labelsize=xt_labelsize)

    if axes.size > 0:
        axes[0, 0].set_xlim(0, max_lag)
        axes[0, 0].set_ylim(-1, 1)

    return axes
