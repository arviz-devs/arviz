"""Matplotlib ELPDPlot."""
import warnings

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ....rcparams import rcParams
from ...plot_utils import _scale_fig_size, color_from_dim, set_xticklabels
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


def plot_elpd(
    ax,
    models,
    pointwise_data,
    numvars,
    figsize,
    textsize,
    plot_kwargs,
    xlabels,
    coord_labels,
    xdata,
    threshold,
    legend,
    color,
    backend_kwargs,
    show,
):
    """Matplotlib elpd plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    backend_kwargs.setdefault("constrained_layout", not xlabels)

    plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "scatter")

    markersize = None

    if isinstance(color, str) and color in pointwise_data[0].dims:
        colors, color_mapping = color_from_dim(pointwise_data[0], color)
        cmap_name = plot_kwargs.pop("cmap", plt.rcParams["image.cmap"])
        markersize = plot_kwargs.pop("s", plt.rcParams["lines.markersize"])
        cmap = getattr(cm, cmap_name)
        handles = [
            Line2D([], [], color=cmap(float_color), label=coord, ms=markersize, lw=0, **plot_kwargs)
            for coord, float_color in color_mapping.items()
        ]
        plot_kwargs.setdefault("cmap", cmap_name)
        plot_kwargs.setdefault("s", markersize**2)
        plot_kwargs.setdefault("c", colors)
    else:
        legend = False
    plot_kwargs.setdefault("c", color)

    # flatten data (data must be flattened after selecting, labeling and coloring)
    pointwise_data = [pointwise.values.flatten() for pointwise in pointwise_data]

    if numvars == 2:
        (figsize, ax_labelsize, titlesize, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        plot_kwargs.setdefault("s", markersize**2)
        backend_kwargs.setdefault("figsize", figsize)
        backend_kwargs["squeeze"] = True
        if ax is None:
            fig, ax = create_axes_grid(
                1,
                backend_kwargs=backend_kwargs,
            )
        else:
            fig = ax.get_figure()

        ydata = pointwise_data[0] - pointwise_data[1]
        ax.scatter(xdata, ydata, **plot_kwargs)
        if threshold is not None:
            diff_abs = np.abs(ydata - ydata.mean())
            bool_ary = diff_abs > threshold * ydata.std()
            if coord_labels is None:
                coord_labels = xdata.astype(str)
            outliers = np.nonzero(bool_ary)[0]
            for outlier in outliers:
                label = coord_labels[outlier]
                ax.text(
                    outlier,
                    ydata[outlier],
                    label,
                    horizontalalignment="center",
                    verticalalignment="bottom" if ydata[outlier] > 0 else "top",
                    fontsize=0.8 * xt_labelsize,
                )

        ax.set_title("{} - {}".format(*models), fontsize=titlesize, wrap=True)
        ax.set_ylabel("ELPD difference", fontsize=ax_labelsize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)
        if xlabels:
            set_xticklabels(ax, coord_labels)
            fig.autofmt_xdate()
            fig.tight_layout()
        if legend:
            ncols = len(handles) // 6 + 1
            ax.legend(handles=handles, ncol=ncols, title=color)

    else:
        max_plots = (
            numvars**2 if rcParams["plot.max_subplots"] is None else rcParams["plot.max_subplots"]
        )
        vars_to_plot = np.sum(np.arange(numvars).cumsum() < max_plots)
        if vars_to_plot < numvars:
            warnings.warn(
                "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
                "of resulting ELPD pairwise plots with these variables, generating only a "
                "{side}x{side} grid".format(max_plots=max_plots, side=vars_to_plot),
                UserWarning,
            )
            numvars = vars_to_plot

        (figsize, ax_labelsize, titlesize, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )
        plot_kwargs.setdefault("s", markersize**2)

        if ax is None:
            fig, ax = plt.subplots(
                numvars - 1,
                numvars - 1,
                figsize=figsize,
                squeeze=False,
                constrained_layout=not xlabels,
                sharey="row",
                sharex="col",
            )
        else:
            fig = ax.ravel()[0].get_figure()

        for i in range(0, numvars - 1):
            var1 = pointwise_data[i]

            for j in range(0, numvars - 1):
                if j < i:
                    ax[j, i].axis("off")
                    continue

                var2 = pointwise_data[j + 1]
                ax[j, i].scatter(xdata, var1 - var2, **plot_kwargs)
                if threshold is not None:
                    ydata = var1 - var2
                    diff_abs = np.abs(ydata - ydata.mean())
                    bool_ary = diff_abs > threshold * ydata.std()
                    if coord_labels is None:
                        coord_labels = xdata.astype(str)
                    outliers = np.nonzero(bool_ary)[0]
                    for outlier in outliers:
                        label = coord_labels[outlier]
                        ax[j, i].text(
                            outlier,
                            ydata[outlier],
                            label,
                            horizontalalignment="center",
                            verticalalignment="bottom" if ydata[outlier] > 0 else "top",
                            fontsize=0.8 * xt_labelsize,
                        )

                if i == 0:
                    ax[j, i].set_ylabel("ELPD difference", fontsize=ax_labelsize, wrap=True)

                ax[j, i].tick_params(labelsize=xt_labelsize)
                ax[j, i].set_title(f"{models[i]} - {models[j + 1]}", fontsize=titlesize, wrap=True)
        if xlabels:
            for i in range(len(ax)):
                set_xticklabels(ax[-1, i], coord_labels)
            fig.autofmt_xdate()
            fig.tight_layout()
        if legend:
            ncols = len(handles) // 6 + 1
            ax[0, 1].legend(
                handles=handles, ncol=ncols, title=color, bbox_to_anchor=(0, 1), loc="upper left"
            )

    if backend_show(show):
        plt.show()

    return ax
