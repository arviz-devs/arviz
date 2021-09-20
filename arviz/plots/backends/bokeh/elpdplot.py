"""Bokeh ELPDPlot."""
import warnings

import bokeh.plotting as bkp
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Title
from bokeh.models.glyphs import Scatter
from ....rcparams import _validate_bokeh_marker, rcParams
from ...plot_utils import _scale_fig_size, color_from_dim, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


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
    legend,  # pylint: disable=unused-argument
    color,
    backend_kwargs,
    show,
):
    """Bokeh elpd plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    plot_kwargs.setdefault("marker", rcParams["plot.bokeh.marker"])
    if isinstance(color, str):
        if color in pointwise_data[0].dims:
            colors, _ = color_from_dim(pointwise_data[0], color)
            plot_kwargs.setdefault("color", vectorized_to_hex(colors))
    plot_kwargs.setdefault("color", vectorized_to_hex(color))

    # flatten data (data must be flattened after selecting, labeling and coloring)
    pointwise_data = [pointwise.values.flatten() for pointwise in pointwise_data]

    if numvars == 2:
        (figsize, _, _, _, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        plot_kwargs.setdefault("s", markersize)

        if ax is None:
            ax = create_axes_grid(
                1,
                figsize=figsize,
                squeeze=True,
                backend_kwargs=backend_kwargs,
            )
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
            dpi = backend_kwargs.pop("dpi")
            ax = []
            for row in range(numvars - 1):
                ax_row = []
                for col in range(numvars - 1):
                    if row == 0 and col == 0:
                        ax_first = bkp.figure(
                            width=int(figsize[0] / (numvars - 1) * dpi),
                            height=int(figsize[1] / (numvars - 1) * dpi),
                            **backend_kwargs,
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
                                **backend_kwargs,
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
    sizes = np.ones(len(xdata)) * plot_kwargs.get("s")
    glyph = Scatter(
        x="xdata",
        y="ydata",
        size="sizes",
        line_color=plot_kwargs.get("color", "black"),
        marker=marker,
    )
    source = ColumnDataSource(dict(xdata=xdata, ydata=ydata, sizes=sizes))
    ax_.add_glyph(source, glyph)
    if threshold is not None:
        diff_abs = np.abs(ydata - ydata.mean())
        bool_ary = diff_abs > threshold * ydata.std()
        if coord_labels is None:
            coord_labels = xdata.astype(str)
        outliers = np.nonzero(bool_ary)[0]
        for outlier in outliers:
            label = coord_labels[outlier]
            ax_.text(
                x=[outlier],
                y=[ydata[outlier]],
                text=label,
                text_color="black",
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
    title.text = f"{model1} - {model2}"
    ax_.title = title
