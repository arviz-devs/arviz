"""Bokeh ELPDPlot."""
import warnings

import bokeh.plotting as bkp
from bokeh.models.annotations import Title
from bokeh.models import ColumnDataSource
import bokeh.models.markers as mk
import numpy as np

from . import backend_kwarg_defaults
from .. import show_layout
from ...plot_utils import _scale_fig_size, color_from_dim
from ....rcparams import rcParams, _validate_bokeh_marker


def plot_elpd(
    ax,
    models,
    pointwise_data,
    numvars,
    figsize,
    textsize,
    plot_kwargs,
    color,
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
        **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi"),),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")

    plot_kwargs.setdefault("marker", rcParams["plot.bokeh.marker"])
    plot_kwargs.setdefault("color", color)
    if isinstance(color, str):
        if color in pointwise_data[0].dims:
            colors, _ = color_from_dim(pointwise_data[0], color)

    if numvars == 2:
        (figsize, _, _, _, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        plot_kwargs.setdefault("s", markersize)

        if ax is None:
            backend_kwargs.setdefault("width", int(figsize[0] * dpi))
            backend_kwargs.setdefault("height", int(figsize[1] * dpi))
            ax = bkp.figure(**backend_kwargs)
        ydata = pointwise_data[0] - pointwise_data[1]
        _plot_atomic_elpd(
            ax, xdata, ydata, *models, threshold, coord_labels, xlabels, True, True, plot_kwargs
        )

        show_layout(ax, show)

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
                UserWarning,
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

        show_layout(ax, show)

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
    marker = _validate_bokeh_marker(plot_kwargs.get("marker"))
    marker_func = getattr(mk, marker)
    sizes = np.ones(len(xdata)) * plot_kwargs.get("s")
    glyph = marker_func(
        x="xdata", y="ydata", size="sizes", line_color=plot_kwargs.get("color", "black")
    )
    source = ColumnDataSource(dict(xdata=xdata, ydata=ydata, sizes=sizes))
    ax_.add_glyph(source, glyph)
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
