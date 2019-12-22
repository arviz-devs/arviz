"""Bokeh pairplot."""

import warnings
from uuid import uuid4

import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource

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
    divergences,
    diverging_mask,
    flat_var_names,
    backend_kwargs,
    show,
):
    """Bokeh pair plot."""
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
        (figsize, _, _, _, _, _) = _scale_fig_size(figsize, textsize, numvars - 1, numvars - 1)

        if ax is None:
            ax = bkp.figure(
                width=int(figsize[0] * dpi), height=int(figsize[1] * dpi), **backend_kwargs
            )

        if kind == "scatter":
            ax.circle(_posterior[0], _posterior[1])
        elif kind == "kde":
            plot_kde(
                _posterior[0],
                _posterior[1],
                contour=contour,
                fill_last=fill_last,
                ax=ax,
                backend="bokeh",
                backend_kwargs={},
                show=False,
            )
        else:
            ax.hexbin(_posterior[0], _posterior[1], size=0.5)
            ax.grid.visible = False

        if divergences:
            ax.circle(
                _posterior[0][diverging_mask],
                _posterior[1][diverging_mask],
                line_color="black",
                fill_color="orange",
                line_width=1,
                size=6,
            )

        ax.xaxis.axis_label = flat_var_names[0]
        ax.yaxis.axis_label = flat_var_names[1]

        if backend_show(show):
            bkp.show(ax)

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

        (figsize, _, _, _, _, _) = _scale_fig_size(figsize, textsize, numvars - 2, numvars - 2)

        if ax is None:
            ax = []
            for row in range(numvars - 1):
                row_ax = []
                for col in range(numvars - 1):
                    if row < col:
                        row_ax.append(None)
                    else:
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
                        ax_ = bkp.figure(
                            width=int(figsize[0] / (numvars - 1) * 60),
                            height=int(figsize[1] / (numvars - 1) * 60),
                            output_backend="webgl",
                            tools=tools,
                        )
                        row_ax.append(ax_)
                ax.append(row_ax)
            ax = np.array(ax)

        tmp_flat_var_names = None
        if len(flat_var_names) == len(list(set(flat_var_names))):
            source = ColumnDataSource(
                data=dict(zip(flat_var_names, [list(post) for post in _posterior]))
            )
        else:
            tmp_flat_var_names = ["{}__{}".format(name, str(uuid4())) for name in flat_var_names]
            source = ColumnDataSource(
                data=dict(zip(tmp_flat_var_names, [list(post) for post in _posterior]))
            )

        for i in range(0, numvars - 1):
            var1 = flat_var_names[i] if tmp_flat_var_names is None else tmp_flat_var_names[i]

            for j in range(0, numvars - 1):
                if j < i:
                    continue

                var2 = (
                    flat_var_names[j + 1]
                    if tmp_flat_var_names is None
                    else tmp_flat_var_names[j + 1]
                )

                if kind == "scatter":
                    ax[j, i].circle(var1, var2, source=source)

                elif kind == "kde":
                    var1 = _posterior[i]
                    var2 = _posterior[j + 1]
                    plot_kde(
                        var1,
                        var2,
                        contour=contour,
                        fill_last=fill_last,
                        ax=ax[j, i],
                        backend="bokeh",
                        backend_kwargs={},
                        show=False,
                        **plot_kwargs
                    )

                else:
                    var1_ = _posterior[i]
                    var2_ = _posterior[j + 1]
                    ax[j, i].grid.visible = False
                    ax[j, i].hexbin(var1_, var2_, size=0.5)

                if divergences:
                    var1_ = _posterior[i]
                    var2_ = _posterior[j + 1]
                    ax[j, i].circle(
                        var1_[diverging_mask],
                        var2_[diverging_mask],
                        line_color="black",
                        fill_color="orange",
                        line_width=1,
                        size=10,
                    )

                ax[j, i].xaxis.axis_label = flat_var_names[i]
                ax[j, i].yaxis.axis_label = flat_var_names[j + 1]

        if backend_show(show):
            grid = gridplot(ax.tolist(), toolbar_location="above")
            bkp.show(grid)

    return ax
