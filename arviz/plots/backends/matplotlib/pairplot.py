"""Matplotlib pairplot."""
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ....rcparams import rcParams
from ...distplot import plot_dist
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size, calculate_point_estimate
from . import backend_kwarg_defaults, backend_show, matplotlib_kwarg_dealiaser


def plot_pair(
    ax,
    plotters,
    numvars,
    figsize,
    textsize,
    kind,
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
    marginals,
    point_estimate,
    point_estimate_kwargs,
    point_estimate_marker_kwargs,
    reference_values,
    reference_values_kwargs,
):
    """Matplotlib pairplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    scatter_kwargs = matplotlib_kwarg_dealiaser(scatter_kwargs, "scatter")

    scatter_kwargs.setdefault("marker", ".")
    scatter_kwargs.setdefault("lw", 0)
    # Sets the default zorder higher than zorder of grid, which is 0.5
    scatter_kwargs.setdefault("zorder", 0.6)

    if kde_kwargs is None:
        kde_kwargs = {}

    if hexbin_kwargs is None:
        hexbin_kwargs = {}
    hexbin_kwargs.setdefault("mincnt", 1)

    divergences_kwargs = matplotlib_kwarg_dealiaser(divergences_kwargs, "plot")
    divergences_kwargs.setdefault("marker", "o")
    divergences_kwargs.setdefault("markeredgecolor", "k")
    divergences_kwargs.setdefault("color", "C1")
    divergences_kwargs.setdefault("lw", 0)

    if marginal_kwargs is None:
        marginal_kwargs = {}

    point_estimate_kwargs = matplotlib_kwarg_dealiaser(point_estimate_kwargs, "fill_between")
    point_estimate_kwargs.setdefault("color", "k")

    if kind != "kde":
        kde_kwargs.setdefault("contourf_kwargs", {})
        kde_kwargs["contourf_kwargs"].setdefault("alpha", 0)
        kde_kwargs.setdefault("contour_kwargs", {})
        kde_kwargs["contour_kwargs"].setdefault("colors", "k")

    if reference_values:
        reference_values_copy = {}
        label = []
        for variable in list(reference_values.keys()):
            if " " in variable:
                variable_copy = variable.replace(" ", "\n", 1)
            else:
                variable_copy = variable

            label.append(variable_copy)
            reference_values_copy[variable_copy] = reference_values[variable]

        difference = set(flat_var_names).difference(set(label))

        if difference:
            warn = [diff.replace("\n", " ", 1) for diff in difference]
            warnings.warn(
                "Argument reference_values does not include reference value for: {}".format(
                    ", ".join(warn)
                ),
                UserWarning,
            )

    reference_values_kwargs = matplotlib_kwarg_dealiaser(reference_values_kwargs, "plot")

    reference_values_kwargs.setdefault("color", "C2")
    reference_values_kwargs.setdefault("markeredgecolor", "k")
    reference_values_kwargs.setdefault("marker", "o")

    point_estimate_marker_kwargs = matplotlib_kwarg_dealiaser(
        point_estimate_marker_kwargs, "scatter"
    )
    point_estimate_marker_kwargs.setdefault("marker", "s")
    point_estimate_marker_kwargs.setdefault("color", "k")

    # pylint: disable=too-many-nested-blocks
    if numvars == 2:
        (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        backend_kwargs.setdefault("figsize", figsize)

        marginal_kwargs.setdefault("plot_kwargs", {})
        marginal_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)

        point_estimate_marker_kwargs.setdefault("s", markersize + 50)

        # Flatten data
        x = plotters[0][-1].flatten()
        y = plotters[1][-1].flatten()
        if ax is None:
            if marginals:
                # Instantiate figure and grid
                widths = [2, 2, 2, 1]
                heights = [1.4, 2, 2, 2]
                fig = plt.figure(**backend_kwargs)
                grid = plt.GridSpec(
                    4,
                    4,
                    hspace=0.1,
                    wspace=0.1,
                    figure=fig,
                    width_ratios=widths,
                    height_ratios=heights,
                )
                # Set up main plot
                ax = fig.add_subplot(grid[1:, :-1])
                # Set up top KDE
                ax_hist_x = fig.add_subplot(grid[0, :-1], sharex=ax)
                ax_hist_x.set_yticks([])
                # Set up right KDE
                ax_hist_y = fig.add_subplot(grid[1:, -1], sharey=ax)
                ax_hist_y.set_xticks([])
                ax_return = np.array([[ax_hist_x, None], [ax, ax_hist_y]])

                for val, ax_, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
                    plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax_, **marginal_kwargs)

                # Personalize axes
                ax_hist_x.tick_params(labelleft=False, labelbottom=False)
                ax_hist_y.tick_params(labelleft=False, labelbottom=False)
            else:
                fig, ax = plt.subplots(numvars - 1, numvars - 1, **backend_kwargs)
        else:
            if marginals:
                assert ax.shape == (numvars, numvars)
                if ax[0, 1] is not None and ax[0, 1].get_figure() is not None:
                    ax[0, 1].remove()
                ax_return = ax
                ax_hist_x = ax[0, 0]
                ax_hist_y = ax[1, 1]
                ax = ax[1, 0]
                for val, ax_, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
                    plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax_, **marginal_kwargs)
            else:
                ax = np.atleast_2d(ax)[0, 0]

        if "scatter" in kind:
            ax.plot(x, y, **scatter_kwargs)
        if "kde" in kind:
            plot_kde(x, y, ax=ax, **kde_kwargs)
        if "hexbin" in kind:
            hexbin = ax.hexbin(
                x,
                y,
                gridsize=gridsize,
                **hexbin_kwargs,
            )
            ax.grid(False)

        if kind == "hexbin" and colorbar:
            cbar = ax.figure.colorbar(hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], ax=ax)
            cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

        if divergences:
            ax.plot(
                x[diverging_mask],
                y[diverging_mask],
                **divergences_kwargs,
            )

        if point_estimate:
            pe_x = calculate_point_estimate(point_estimate, x)
            pe_y = calculate_point_estimate(point_estimate, y)
            if marginals:
                ax_hist_x.axvline(pe_x, **point_estimate_kwargs)
                ax_hist_y.axhline(pe_y, **point_estimate_kwargs)

            ax.axvline(pe_x, **point_estimate_kwargs)
            ax.axhline(pe_y, **point_estimate_kwargs)

            ax.scatter(pe_x, pe_y, **point_estimate_marker_kwargs)

        if reference_values:
            ax.plot(
                reference_values_copy[flat_var_names[0]],
                reference_values_copy[flat_var_names[1]],
                **reference_values_kwargs,
            )
        ax.set_xlabel(f"{flat_var_names[0]}", fontsize=ax_labelsize, wrap=True)
        ax.set_ylabel(f"{flat_var_names[1]}", fontsize=ax_labelsize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)

    else:
        not_marginals = int(not marginals)
        num_subplot_cols = numvars - not_marginals
        max_plots = (
            num_subplot_cols ** 2
            if rcParams["plot.max_subplots"] is None
            else rcParams["plot.max_subplots"]
        )
        cols_to_plot = np.sum(np.arange(1, num_subplot_cols + 1).cumsum() <= max_plots)
        if cols_to_plot < num_subplot_cols:
            vars_to_plot = cols_to_plot
            warnings.warn(
                "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
                "of resulting pair plots with these variables, generating only a "
                "{side}x{side} grid".format(max_plots=max_plots, side=vars_to_plot),
                UserWarning,
            )
        else:
            vars_to_plot = numvars - not_marginals

        (figsize, ax_labelsize, _, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, vars_to_plot, vars_to_plot
        )
        backend_kwargs.setdefault("figsize", figsize)
        point_estimate_marker_kwargs.setdefault("s", markersize + 50)

        if ax is None:
            fig, ax = plt.subplots(
                vars_to_plot,
                vars_to_plot,
                **backend_kwargs,
            )
        hexbin_values = []
        for i in range(0, vars_to_plot):
            var1 = plotters[i][-1].flatten()

            for j in range(0, vars_to_plot):
                var2 = plotters[j + not_marginals][-1].flatten()
                if i > j:
                    if ax[j, i].get_figure() is not None:
                        ax[j, i].remove()
                    continue

                elif i == j and marginals:
                    loc = "right"
                    plot_dist(var1, ax=ax[i, j], **marginal_kwargs)

                else:
                    if i == j:
                        loc = "left"

                    if "scatter" in kind:
                        ax[j, i].plot(var1, var2, **scatter_kwargs)

                    if "kde" in kind:

                        plot_kde(
                            var1,
                            var2,
                            ax=ax[j, i],
                            **deepcopy(kde_kwargs),
                        )

                    if "hexbin" in kind:
                        ax[j, i].grid(False)
                        hexbin = ax[j, i].hexbin(var1, var2, gridsize=gridsize, **hexbin_kwargs)

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

                        if marginals:
                            ax[j - 1, i].axvline(pe_x, **point_estimate_kwargs)
                            pe_last = calculate_point_estimate(point_estimate, plotters[-1][-1])
                            ax[-1, -1].axvline(pe_last, **point_estimate_kwargs)

                        ax[j, i].scatter(pe_x, pe_y, **point_estimate_marker_kwargs)

                    if reference_values:
                        x_name = flat_var_names[i]
                        y_name = flat_var_names[j + not_marginals]
                        if x_name and y_name not in difference:
                            ax[j, i].plot(
                                reference_values_copy[x_name],
                                reference_values_copy[y_name],
                                **reference_values_kwargs,
                            )

                if j != vars_to_plot - 1:
                    ax[j, i].axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_xlabel(f"{flat_var_names[i]}", fontsize=ax_labelsize, wrap=True)
                if i != 0:
                    ax[j, i].axes.get_yaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_ylabel(
                        f"{flat_var_names[j + not_marginals]}",
                        fontsize=ax_labelsize,
                        wrap=True,
                    )
                ax[j, i].tick_params(labelsize=xt_labelsize)

    if backend_show(show):
        plt.show()

    if marginals and numvars == 2:
        return ax_return
    return ax
