"""Matplotlib pairplot."""

import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ...kdeplot import plot_kde
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size
from ....rcparams import rcParams


def plot_pair(
    ax,
    infdata_group,
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
    marginal_kwargs,
    show,
    diagonal,
):
    """Matplotlib pairplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    backend_kwargs.pop("constrained_layout")

    if numvars == 2:
        (figsize, ax_labelsize, _, xt_labelsize, linewidth, _) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )

        if marginal_kwargs is None:
            marginal_kwargs = {}

        marginal_kwargs.setdefault("plot_kwargs", {})
        marginal_kwargs["plot_kwargs"]["linewidth"] = linewidth

        if ax is None and not diagonal:
            fig, axjoin = plt.subplots(figsize=figsize, **backend_kwargs)

        elif ax is None and diagonal:
            # Instantiate figure and grid
            fig, _ = plt.subplots(0, 0, figsize=figsize, **backend_kwargs)
            grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1, figure=fig)
            # Set up main plot
            axjoin = fig.add_subplot(grid[1:, :-1])
            # Set up top KDE
            ax_hist_x = fig.add_subplot(grid[0, :-1], sharex=axjoin)
            # Set up right KDE
            ax_hist_y = fig.add_subplot(grid[1:, -1], sharey=axjoin)
            # Flatten data
            x = infdata_group[0].flatten()
            y = infdata_group[1].flatten()

            for val, ax_, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
                plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax_, **marginal_kwargs)

            ax_hist_x.set_xlim(axjoin.get_xlim())
            ax_hist_y.set_ylim(axjoin.get_ylim())

            # Personalize axes
            ax_hist_x.tick_params(labelleft=False, labelbottom=False)
            ax_hist_y.tick_params(labelleft=False, labelbottom=False)
        else:
            axjoin = ax

        if kind == "scatter":
            axjoin.plot(infdata_group[0], infdata_group[1], **plot_kwargs)
        elif kind == "kde":
            plot_kde(
                infdata_group[0],
                infdata_group[1],
                contour=contour,
                fill_last=fill_last,
                ax=axjoin,
                **plot_kwargs
            )
        elif kind == "hexbin":
            hexbin = axjoin.hexbin(
                infdata_group[0], infdata_group[1], mincnt=1, gridsize=gridsize, **plot_kwargs
            )
            axjoin.grid(False)

        else:
            axjoin.plot(infdata_group[0], infdata_group[1], zorder=-1, **plot_kwargs)
            plot_kde(
                infdata_group[0],
                infdata_group[1],
                ax=axjoin,
                contourf_kwargs={"alpha": 0},
                contour_kwargs={"colors": "k"},
                fill_last=False,
            )
        if kind == "hexbin" and colorbar:
            cbar = axjoin.figure.colorbar(hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], ax=ax)
            cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

        if divergences:
            axjoin.plot(
                infdata_group[0][diverging_mask],
                infdata_group[1][diverging_mask],
                **divergences_kwargs
            )

        axjoin.set_xlabel("{}".format(flat_var_names[0]), fontsize=ax_labelsize, wrap=True)
        axjoin.set_ylabel("{}".format(flat_var_names[1]), fontsize=ax_labelsize, wrap=True)
        axjoin.tick_params(labelsize=xt_labelsize)

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
                UserWarning,
            )
            numvars = vars_to_plot

        (figsize, ax_labelsize, _, xt_labelsize, _, _) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )

        if ax is None:
            fig, ax = plt.subplots(numvars, numvars, figsize=figsize, **backend_kwargs)
        hexbin_values = []
        for i in range(0, numvars):
            var1 = infdata_group[i]

            for j in range(0, numvars):
                if i > j:
                    # ax[j, i].axis("off")
                    ax[j, i].remove()
                    continue

                var2 = infdata_group[j]

                if i == j:
                    if diagonal:
                        loc = "right"
                        plot_dist(var1, ax=ax[i, j], **marginal_kwargs)
                    else:
                        loc = "left"
                        ax[j, i].remove()  # .axis("off")
                        continue
                if i < j:
                    if kind == "scatter":
                        ax[j, i].plot(var1, var2, **plot_kwargs)

                    elif kind == "kde":
                        plot_kde(
                            var1,
                            var2,
                            contour=contour,
                            fill_last=fill_last,
                            ax=ax[j, i],
                            **plot_kwargs
                        )

                    elif kind == "hexbin":
                        ax[j, i].grid(False)
                        hexbin = ax[j, i].hexbin(
                            var1, var2, mincnt=1, gridsize=gridsize, **plot_kwargs
                        )

                    else:
                        ax[j, i].plot(var1, var2, **plot_kwargs, zorder=-1)
                        plot_kde(
                            var1,
                            var2,
                            ax=ax[j, i],
                            contourf_kwargs={"alpha": 0},
                            contour_kwargs={"colors": "k"},
                            fill_last=False,
                        )

                    if divergences:
                        ax[j, i].plot(
                            var1[diverging_mask], var2[diverging_mask], **divergences_kwargs
                        )

                    if kind == "hexbin" and colorbar:
                        hexbin_values.append(hexbin.norm.vmin)
                        hexbin_values.append(hexbin.norm.vmax)
                        # if j == i == 0 and colorbar:
                        divider = make_axes_locatable(ax[-1, -1])
                        cax = divider.append_axes(loc, size="7%")
                        cbar = fig.colorbar(
                            hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], cax=cax
                        )
                        cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

                if j != numvars - 1:
                    ax[j, i].axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_xlabel(
                        "{}".format(flat_var_names[i]), fontsize=ax_labelsize, wrap=True
                    )
                if i != 0:
                    ax[j, i].axes.get_yaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_ylabel(
                        "{}".format(flat_var_names[j]), fontsize=ax_labelsize, wrap=True
                    )

                ax[j, i].tick_params(labelsize=xt_labelsize)
    if backend_show(show):
        plt.show()

    return ax
