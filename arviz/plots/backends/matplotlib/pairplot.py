"""Matplotlib pairplot."""

import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size
from ....rcparams import rcParams


def plot_pair(
    ax,
    _posterior,
    numvars,
    figsize,
    textsize,
    kind,
    plot_kwargs,
    contour,
    fill_last,
    gridsize,
    colorbar,
    divergences,
    diverging_mask,
    divergences_kwargs,
    flat_var_names,
    backend_kwargs,
    show,
):
    """Matplotlib pairplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    if numvars == 2:
        (figsize, ax_labelsize, _, xt_labelsize, _, _) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **backend_kwargs)

        if kind == "scatter":
            ax.plot(_posterior[0], _posterior[1], **plot_kwargs)
        elif kind == "kde":
            plot_kde(
                _posterior[0],
                _posterior[1],
                contour=contour,
                fill_last=fill_last,
                ax=ax,
                **plot_kwargs
            )
        else:
            hexbin = ax.hexbin(
                _posterior[0], _posterior[1], mincnt=1, gridsize=gridsize, **plot_kwargs
            )
            ax.grid(False)

        if kind == "hexbin" and colorbar:
            cbar = ax.figure.colorbar(hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], ax=ax)
            cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

        if divergences:
            ax.plot(
                _posterior[0][diverging_mask], _posterior[1][diverging_mask], **divergences_kwargs
            )

        ax.set_xlabel("{}".format(flat_var_names[0]), fontsize=ax_labelsize, wrap=True)
        ax.set_ylabel("{}".format(flat_var_names[1]), fontsize=ax_labelsize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)

    else:
        max_plots = (
            numvars ** 2 if rcParams["plot.max_subplots"] is None else rcParams["plot.max_subplots"]
        )
        vars_to_plot = np.sum(np.arange(numvars).cumsum() < max_plots)
        if vars_to_plot < numvars:
            warnings.warn(
                "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
                "of resulting pair plots with these variables, generating only a "
                "{side}x{side} grid".format(max_plots=max_plots, side=vars_to_plot),
                SyntaxWarning,
            )
            numvars = vars_to_plot

        (figsize, ax_labelsize, _, xt_labelsize, _, _) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )

        if ax is None:
            fig, ax = plt.subplots(numvars - 1, numvars - 1, figsize=figsize, **backend_kwargs)
        hexbin_values = []
        for i in range(0, numvars - 1):
            var1 = _posterior[i]

            for j in range(0, numvars - 1):
                if j < i:
                    ax[j, i].axis("off")
                    continue

                var2 = _posterior[j + 1]

                if kind == "scatter":
                    ax[j, i].plot(var1, var2, **plot_kwargs)

                elif kind == "kde":
                    plot_kde(
                        var1, var2, contour=contour, fill_last=fill_last, ax=ax[j, i], **plot_kwargs
                    )

                else:
                    ax[j, i].grid(False)
                    hexbin = ax[j, i].hexbin(var1, var2, mincnt=1, gridsize=gridsize, **plot_kwargs)
                if kind == "hexbin" and colorbar:
                    hexbin_values.append(hexbin.norm.vmin)
                    hexbin_values.append(hexbin.norm.vmax)
                    if j == i == 0 and colorbar:
                        divider = make_axes_locatable(ax[0, 1])
                        cax = divider.append_axes("left", size="7%")
                        cbar = fig.colorbar(
                            hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], cax=cax
                        )
                        cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

                if divergences:
                    ax[j, i].plot(var1[diverging_mask], var2[diverging_mask], **divergences_kwargs)

                if j + 1 != numvars - 1:
                    ax[j, i].axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_xlabel(
                        "{}".format(flat_var_names[i]), fontsize=ax_labelsize, wrap=True
                    )
                if i != 0:
                    ax[j, i].axes.get_yaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_ylabel(
                        "{}".format(flat_var_names[j + 1]), fontsize=ax_labelsize, wrap=True
                    )

                ax[j, i].tick_params(labelsize=xt_labelsize)

    if backend_show(show):
        plt.show()

    return ax
