"""Matplotlib ELPDPlot."""
import warnings

import matplotlib.pyplot as plt
import numpy as np


from . import backend_kwarg_defaults, backend_show
from ...plot_utils import (
    _scale_fig_size,
    set_xticklabels,
)
from ....rcparams import rcParams


def plot_elpd(
    ax,
    models,
    pointwise_data,
    numvars,
    figsize,
    textsize,
    plot_kwargs,
    markersize,
    xlabels,
    coord_labels,
    xdata,
    threshold,
    legend,
    handles,
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
    backend_kwargs["constrained_layout"] = not xlabels

    if numvars == 2:
        (figsize, ax_labelsize, titlesize, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        plot_kwargs.setdefault("s", markersize ** 2)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **backend_kwargs)

        ydata = pointwise_data[0] - pointwise_data[1]
        ax.scatter(xdata, ydata, **plot_kwargs)
        if threshold is not None:
            diff_abs = np.abs(ydata - ydata.mean())
            bool_ary = diff_abs > threshold * ydata.std()
            if coord_labels is None:
                coord_labels = xdata.astype(str)
            outliers = np.argwhere(bool_ary).squeeze()
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
            numvars ** 2 if rcParams["plot.max_subplots"] is None else rcParams["plot.max_subplots"]
        )
        vars_to_plot = np.sum(np.arange(numvars).cumsum() < max_plots)
        if vars_to_plot < numvars:
            warnings.warn(
                "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
                "of resulting ELPD pairwise plots with these variables, generating only a "
                "{side}x{side} grid".format(max_plots=max_plots, side=vars_to_plot),
                SyntaxWarning,
            )
            numvars = vars_to_plot

        (figsize, ax_labelsize, titlesize, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )
        plot_kwargs.setdefault("s", markersize ** 2)

        if ax is None:
            fig, ax = plt.subplots(
                numvars - 1,
                numvars - 1,
                figsize=figsize,
                constrained_layout=not xlabels,
                sharey="row",
                sharex="all",
            )

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
                    outliers = np.argwhere(bool_ary).squeeze()
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
                ax[j, i].set_title(
                    "{} - {}".format(models[i], models[j + 1]), fontsize=titlesize, wrap=True
                )
        if xlabels:
            set_xticklabels(ax[-1, -1], coord_labels)
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
