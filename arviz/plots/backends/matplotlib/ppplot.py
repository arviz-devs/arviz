"""Matplotib Prior Posterior plot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ...kdeplot import plot_kde
from . import backend_kwarg_defaults


def plot_pp(
    ax,
    nvars,
    ngroups,
    figsize,
    pp_plotters,
    legend,
    groups,
    prior_kwargs,
    posterior_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib pp plot."""

    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}
    if ax is None:
        axes = np.empty((nvars, ngroups + 1), dtype=object)
        fig = plt.figure(**backend_kwargs, figsize=figsize)
        gs = fig.add_gridspec(ncols=ngroups, nrows=nvars * 2)
        for i in range(nvars):
            for j in range(ngroups):
                axes[i, j] = fig.add_subplot(gs[2 * i, j])
            axes[i, -1] = fig.add_subplot(gs[2 * i + 1, :])

    else:
        axes = ax
        if ax.shape != (nvars, ngroups + 1):
            raise ValueError(
                "Found {} shape of axes, which is not equal to data shape {}.".format(
                    axes.shape, (nvars, ngroups + 1)
                )
            )

    for idx, plotter in enumerate(pp_plotters):
        group = groups[idx]
        kwargs = prior_kwargs if group == "prior" else posterior_kwargs
        for idx2, (var, coord, data,) in enumerate(plotter):
            _pp_helper(
                ax=axes[idx2, idx],
                data=data,
                group=group,
                var_name=var,
                coord_name=coord,
                legend=legend,
                **kwargs,
            )
            _pp_helper(
                ax=axes[idx2, -1],
                data=data,
                group=group,
                var_name=var,
                coord_name=coord,
                legend=legend,
                **kwargs,
            )

    if backend_show(show):
        plt.show()

    return axes


def _pp_helper(ax, data, group, var_name, coord_name, legend, **kwargs):
    plot_kde(
        data,
        label="{} {} {}".format(group, var_name, coord_name) if legend else None,
        ax=ax,
        **kwargs,
    )
