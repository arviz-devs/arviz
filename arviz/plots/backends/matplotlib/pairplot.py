"""Matplotlib pairplot."""

import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ...kdeplot import plot_kde
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size, calculate_point_estimate
from ....rcparams import rcParams


def plot_pair(
    ax,
    infdata_group,
    numvars,
    figsize,
    textsize,
    kind,
    fill_last,
    contour,
    plot_kwargs,
    scatter_kwargs,
    kde_kwargs,
    hexbin_kwargs,
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
    point_estimate,
    point_estimate_kwargs,
    point_estimate_marker_kwargs,
):
    """Matplotlib pairplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    backend_kwargs.pop("constrained_layout")

    if kind != "kde":
        kde_kwargs.setdefault("contourf_kwargs", {"alpha": 0})
        kde_kwargs.setdefault("contour_kwargs", {})
        kde_kwargs["contour_kwargs"].setdefault("colors", "k")

    # pylint: disable=R1702
    if numvars == 2:
        (figsize, ax_labelsize, _, xt_labelsize, linewidth, _) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )

        marginal_kwargs.setdefault("plot_kwargs", {})
        marginal_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)

        # Flatten data
        x = infdata_group[0].flatten()
        y = infdata_group[1].flatten()
        if ax is None:
            if diagonal:
                # Instantiate figure and grid
                fig, _ = plt.subplots(0, 0, figsize=figsize, **backend_kwargs)
                grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1, figure=fig)
                # Set up main plot
                ax = fig.add_subplot(grid[1:, :-1])
                # Set up top KDE
                ax_hist_x = fig.add_subplot(grid[0, :-1], sharex=ax)
                # Set up right KDE
                ax_hist_y = fig.add_subplot(grid[1:, -1], sharey=ax)

                for val, ax_, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
                    plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax_, **marginal_kwargs)

                ax_hist_x.set_xlim(ax.get_xlim())
                ax_hist_y.set_ylim(ax.get_ylim())

                # Personalize axes
                ax_hist_x.tick_params(labelleft=False, labelbottom=False)
                ax_hist_y.tick_params(labelleft=False, labelbottom=False)
            else:
                fig, ax = plt.subplots(numvars - 1, numvars - 1, figsize=figsize, **backend_kwargs)
        else:
            if diagonal:
                assert ax.shape == (numvars, numvars)
                ax[0, 1].remove()
                ax_hist_x = ax[0, 0]
                ax_hist_y = ax[1, 1]
                ax = ax[1, 0]
                for val, ax_, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
                    plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax_, **marginal_kwargs)
            else:
                ax = np.atleast_2d(ax)[0, 0]

        if "scatter" in kind:
            ax.plot(infdata_group[0], infdata_group[1], **scatter_kwargs, **plot_kwargs)
        if "kde" in kind:
            plot_kde(infdata_group[0], infdata_group[1], ax=ax, **kde_kwargs, **plot_kwargs)
        if "hexbin" in kind:
            hexbin = ax.hexbin(
                infdata_group[0],
                infdata_group[1],
                mincnt=1,
                gridsize=gridsize,
                **hexbin_kwargs,
                **plot_kwargs
            )
            ax.grid(False)

        if kind == "hexbin" and colorbar:
            cbar = ax.figure.colorbar(hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], ax=ax)
            cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

        if divergences:
            ax.plot(
                infdata_group[0][diverging_mask],
                infdata_group[1][diverging_mask],
                **divergences_kwargs
            )

        if point_estimate:
            pe_x = calculate_point_estimate(point_estimate, x)
            pe_y = calculate_point_estimate(point_estimate, y)
            if diagonal:
                ax_hist_x.axvline(pe_x, **point_estimate_kwargs)
                ax_hist_y.axhline(pe_y, **point_estimate_kwargs)

            ax.axvline(pe_x, **point_estimate_kwargs)
            ax.axhline(pe_y, **point_estimate_kwargs)

            ax.scatter(pe_x, pe_y, marker="s", s=figsize[0] + 50, **point_estimate_kwargs, zorder=4)

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
                var2 = infdata_group[j]
                if i > j:
                    ax[j, i].remove()
                    continue

                elif i == j:
                    if diagonal:
                        loc = "right"
                        plot_dist(var1, ax=ax[i, j], **marginal_kwargs)
                    else:
                        loc = "left"
                        ax[j, i].remove()
                        continue

                else:
                    if "scatter" in kind:
                        ax[j, i].plot(var1, var2, **scatter_kwargs)

                    if "kde" in kind:

                        plot_kde(
                            var1,
                            var2,
                            ax=ax[j, i],
                            fill_last=fill_last,
                            contour=contour,
                            **kde_kwargs
                        )

                    if "hexbin" in kind:
                        ax[j, i].grid(False)
                        hexbin = ax[j, i].hexbin(
                            var1, var2, mincnt=1, gridsize=gridsize, **hexbin_kwargs
                        )

                    if divergences:
                        ax[j, i].plot(
                            var1[diverging_mask], var2[diverging_mask], **divergences_kwargs
                        )

                    if kind == "hexbin" and colorbar:
                        hexbin_values.append(hexbin.norm.vmin)
                        hexbin_values.append(hexbin.norm.vmax)
                        divider = make_axes_locatable(ax[-1, -1])
                        cax = divider.append_axes(loc, size="7%", pad="5%")
                        cbar = fig.colorbar(
                            hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], cax=cax
                        )
                        cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

                    if point_estimate:
                        pe_x = calculate_point_estimate(point_estimate, var1)
                        pe_y = calculate_point_estimate(point_estimate, var2)
                        ax[j, i].axvline(pe_x, **point_estimate_kwargs)
                        ax[j, i].axhline(pe_y, **point_estimate_kwargs)

                        if diagonal:
                            ax[j - 1, i].axvline(pe_x, **point_estimate_kwargs)
                            pe_last = calculate_point_estimate(point_estimate, infdata_group[-1])
                            ax[-1, -1].axvline(pe_last, **point_estimate_kwargs)

                        ax[j, i].scatter(
                            pe_x, pe_y, s=figsize[0] + 50, zorder=4, **point_estimate_marker_kwargs
                        )

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
