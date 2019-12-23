"""Bokeh ELPDPlot."""
import warnings

import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models.annotations import Title

from . import backend_kwarg_defaults, backend_show
from ...plot_utils import _scale_fig_size
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
    backend_kwargs,
    show,
):
    """Bokeh elpd plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("tools", "plot.bokeh.tools"),
            ("output_backend", "plot.bokeh.output_backend"),
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")
    if numvars == 2:
        (figsize, _, _, _, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        plot_kwargs.setdefault("s", markersize)

        if ax is None:
            ax = bkp.figure(
                width=int(figsize[0] * dpi), height=int(figsize[1] * dpi), **backend_kwargs
            )

        ydata = pointwise_data[0] - pointwise_data[1]
        _plot_atomic_elpd(
            ax, xdata, ydata, *models, threshold, coord_labels, xlabels, True, True, plot_kwargs
        )

        if backend_show(show):
            bkp.show(ax, toolbar_location="above")

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

        (figsize, _, _, _, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )
        plot_kwargs.setdefault("s", markersize)

        if ax is None:
            ax = []
            for row in range(numvars - 1):
                ax_row = []
                for col in range(numvars - 1):
                    if row == 0 and col == 0:
                        ax_first = bkp.figure(
                            width=int(figsize[0] / (numvars - 1) * dpi),
                            height=int(figsize[1] / (numvars - 1) * dpi),
                            **backend_kwargs
                        )
                        ax_row.append(ax_first)
                    elif row < col:
                        ax_row.append(None)
                    else:
                        ax_row.append(
                            bkp.figure(
                                width=int(figsize[0] / (numvars - 1) * dpi),
                                height=int(figsize[1] / (numvars - 1) * dpi),
                                x_range=ax_first.x_range,
                                y_range=ax_first.y_range,
                                **backend_kwargs
                            )
                        )
                ax.append(ax_row)
            ax = np.array(ax)

        for i in range(0, numvars - 1):
            var1 = pointwise_data[i]

            for j in range(0, numvars - 1):
                if j < i:
                    continue

                var2 = pointwise_data[j + 1]
                ydata = var1 - var2
                _plot_atomic_elpd(
                    ax[j, i],
                    xdata,
                    ydata,
                    models[i],
                    models[j + 1],
                    threshold,
                    coord_labels,
                    xlabels,
                    j == numvars - 2,
                    i == 0,
                    plot_kwargs,
                )

        if backend_show(show):
            bkp.show(gridplot(ax.tolist(), toolbar_location="above"))
    return ax


def _plot_atomic_elpd(
    ax_,
    xdata,
    ydata,
    model1,
    model2,
    threshold,
    coord_labels,
    xlabels,
    xlabels_shown,
    ylabels_shown,
    plot_kwargs,
):
    ax_.cross(
        np.asarray(xdata),
        np.asarray(ydata),
        line_color=plot_kwargs.get("color", "black"),
        size=plot_kwargs.get("s"),
    )
    if threshold is not None:
        diff_abs = np.abs(ydata - ydata.mean())
        bool_ary = diff_abs > threshold * ydata.std()
        if coord_labels is None:
            coord_labels = xdata.astype(str)
        outliers = np.argwhere(bool_ary).squeeze()
        for outlier in outliers:
            label = coord_labels[outlier]
            ax_.text(
                x=np.asarray(outlier), y=np.asarray(ydata[outlier]), text=label, text_color="black",
            )
    if ylabels_shown:
        ax_.yaxis.axis_label = "ELPD difference"
    else:
        ax_.yaxis.minor_tick_line_color = None
        ax_.yaxis.major_label_text_font_size = "0pt"

    if xlabels_shown:
        if xlabels:
            ax_.xaxis.ticker = np.arange(0, len(coord_labels))
            ax_.xaxis.major_label_overrides = {
                str(key): str(value)
                for key, value in zip(np.arange(0, len(coord_labels)), list(coord_labels))
            }
    else:
        ax_.xaxis.minor_tick_line_color = None
        ax_.xaxis.major_label_text_font_size = "0pt"
    title = Title()
    title.text = "{} - {}".format(model1, model2)
    ax_.title = title
