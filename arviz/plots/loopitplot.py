import numpy as np
import matplotlib.pyplot as plt

from ..stats import loo_pit as _loo_pit
from .plot_utils import _scale_fig_size
from .kdeplot import _fast_kde


def plot_loo_pit(
    idata,
    y,
    y_hat=None,
    n_unif=100,
    figsize=None,
    textsize=None,
    ax=None,
    plot_kwargs=None,
    plot_unif_kwargs=None,
):
    """Plot LOO-PIT."""
    if plot_kwargs is None:
        plot_kwargs = {}
    if plot_unif_kwargs is None:
        plot_unif_kwargs = {}

    if ax is None:
        (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
            figsize, textsize, 1, 1
        )
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    plot_kwargs.setdefault("color", "C0")
    plot_kwargs.setdefault("linewidth", linewidth)
    plot_unif_kwargs.setdefault("color", "cyan")
    plot_unif_kwargs.setdefault("alpha", 0.5)
    plot_unif_kwargs.setdefault("linewidth", 0.6 * linewidth)
    plot_unif_kwargs.setdefault("zorder", -10)

    loo_pit = _loo_pit(idata, y=y, y_hat=y_hat)
    loo_pit_kde, _, _ = _fast_kde(loo_pit.values.flatten(), xmin=0, xmax=1)

    unif = np.random.uniform(size=(loo_pit.size, n_unif))
    x_vals = np.linspace(0, 1, len(loo_pit_kde))
    for idx in range(n_unif):
        unif_density, _, _ = _fast_kde(unif[:, idx], xmin=0, xmax=1)
        ax.plot(x_vals, unif_density, **plot_unif_kwargs)

    ax.plot(x_vals, loo_pit_kde, **plot_kwargs)
    return ax
