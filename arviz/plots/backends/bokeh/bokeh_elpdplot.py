"""Bokeh ELPDPlot."""
import warnings

import bokeh.plotting as bkp
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
import numpy as np


from ...plot_utils import (
    _scale_fig_size,
    format_coords_as_labels,
)
from ....rcparams import rcParams


def _plot_elpd(
    ax,
    models,
    pointwise_data,
    numvars,
    figsize,
    textsize,
    plot_kwargs,
    markersize,
    xlabels,
    xdata,
    threshold,
    show,
):
    tools = ",".join(
        [
            "pan",
            "wheel_zoom",
            "box_zoom",
            "lasso_select",
            "poly_select",
            "undo",
            "redo",
            "reset",
            "save,hover",
        ]
    )

    if xlabels:
        coord_labels = format_coords_as_labels(pointwise_data[0])

    if numvars == 2:
        (figsize, _, _, _, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        plot_kwargs.setdefault("s", markersize)

        if ax is None:
            ax = bkp.figure(
                width=int(figsize[0] * 60),
                height=int(figsize[1] * 60),
                output_backend="webgl",
                tools=tools,
            )

        ydata = pointwise_data[0] - pointwise_data[1]
        print(np.asarray(xdata).shape)
        print(np.asarray(ydata).shape)
        ax.cross(
            np.asarray(xdata),
            np.asarray(ydata),
            line_color=plot_kwargs.get("color", "black"),
            size=plot_kwargs.get("s"),
        )
        if threshold is not None:
            ydata = ydata.values.flatten()
            diff_abs = np.abs(ydata - ydata.mean())
            bool_ary = diff_abs > threshold * ydata.std()
            try:
                coord_labels
            except NameError:
                coord_labels = xdata.astype(str)
            outliers = np.argwhere(bool_ary).squeeze()
            for outlier in outliers:
                label = coord_labels[outlier]
                ax.text(
                    x=np.asarray(outlier),
                    y=np.asarray(ydata[outlier]),
                    text=label,
                    text_color="black",
                )

        title = Title()
        title.text = "{} - {}".format(*models)
        ax.title = title

        ax.yaxis.axis_label = "ELPD difference"
        if xlabels:
            ax.xaxis.ticker = np.arange(0, len(coord_labels))
            ax.xaxis.major_label_overrides = {
                key: str(value)
                for key, value in zip(np.arange(0, len(coord_labels)), list(coord_labels))
            }

        if show:
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
                            width=int(figsize[0] / (numvars - 1) * 60),
                            height=int(figsize[1] / (numvars - 1) * 60),
                            output_backend="webgl",
                            tools=tools,
                        )
                        ax_row.append(ax_first)
                    elif row < col:
                        ax_row.append(None)
                    else:
                        ax_row.append(
                            bkp.figure(
                                width=int(figsize[0] / (numvars - 1) * 60),
                                height=int(figsize[1] / (numvars - 1) * 60),
                                x_range=ax_first.x_range,
                                y_range=ax_first.y_range,
                                output_backend="webgl",
                                tools=tools,
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
                ax[j, i].cross(
                    np.asarray(xdata),
                    np.asarray(var1 - var2),
                    line_color=plot_kwargs.get("color", "black"),
                    size=plot_kwargs.get("s"),
                )
                if threshold is not None:
                    ydata = (var1 - var2).values.flatten()
                    diff_abs = np.abs(ydata - ydata.mean())
                    bool_ary = diff_abs > threshold * ydata.std()
                    try:
                        coord_labels
                    except NameError:
                        coord_labels = xdata.astype(str)
                    outliers = np.argwhere(bool_ary).squeeze()
                    for outlier in outliers:
                        label = coord_labels[outlier]
                        ax[j, i].text(
                            x=np.asarray(outlier),
                            y=np.asarray(ydata[outlier]),
                            text=label,
                            text_color="black",
                        )

                if i == 0:
                    ax[j, i].yaxis.axis_label = "ELPD difference"

                title = Title()
                title.text = "{} - {}".format(models[i], models[j + 1])
                ax[j, i].title = title

        if xlabels:
            ax[j, i].xaxis.ticker = np.arange(0, len(coord_labels))
            ax[j, i].xaxis.major_label_overrides = {
                key: str(value)
                for key, value in zip(np.arange(0, len(coord_labels)), list(coord_labels))
            }

        if show:
            bkp.show(gridplot([list(item) for item in ax], toolbar_location="above"))
    return ax
