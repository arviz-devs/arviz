"""Bokeh pairplot."""
import warnings
from uuid import uuid4

import bokeh.plotting as bkp
import numpy as np
from bokeh.models import ColumnDataSource, CDSView, GroupFilter, Span

from . import backend_kwarg_defaults
from .. import show_layout
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
    kde_kwargs,
    hexbin_kwargs,
    contour,  # pylint: disable=unused-argument
    plot_kwargs,  # pylint: disable=unused-argument
    fill_last,  # pylint: disable=unused-argument
    divergences,
    diverging_mask,
    flat_var_names,
    backend_kwargs,
    diagonal,
    marginal_kwargs,
    point_estimate,
    point_estimate_kwargs,
    show,
):
    """Bokeh pair plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi"),),
        **backend_kwargs,
    }

    if hexbin_kwargs is None:
        hexbin_kwargs = {}

    hexbin_kwargs.setdefault("size", 0.5)

    if kind != "kde":
        kde_kwargs.setdefault("contourf_kwargs", {"fill_alpha": 0})
        kde_kwargs.setdefault("contour_kwargs", {})
        kde_kwargs["contour_kwargs"].setdefault("line_color", "black")
        kde_kwargs["contour_kwargs"].setdefault("line_alpha", 1)

    dpi = backend_kwargs.pop("dpi")
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

    (figsize, _, _, _, _, _) = _scale_fig_size(figsize, textsize, numvars - 2, numvars - 2)

    tmp_flat_var_names = None
    if len(flat_var_names) == len(list(set(flat_var_names))):
        source_dict = dict(zip(flat_var_names, [list(post) for post in infdata_group]))
    else:
        tmp_flat_var_names = ["{}__{}".format(name, str(uuid4())) for name in flat_var_names]
        source_dict = dict(zip(tmp_flat_var_names, [list(post) for post in infdata_group]))
    if divergences:
        divergenve_name = "divergences_{}".format(str(uuid4()))
        source_dict[divergenve_name] = np.array(diverging_mask).astype(bool).astype(int).astype(str)

    source = ColumnDataSource(data=source_dict)

    if divergences:
        source_nondiv = CDSView(
            source=source, filters=[GroupFilter(column_name=divergenve_name, group="0")]
        )
        source_div = CDSView(
            source=source, filters=[GroupFilter(column_name=divergenve_name, group="1")]
        )

    def get_width_and_height(jointplot, rotate):
        """Compute subplots dimensions for two or more variables."""
        if jointplot:
            if rotate:
                width = int(figsize[0] / (numvars - 1) + 2 * dpi)
                height = int(figsize[1] / (numvars - 1) * dpi)
            else:
                width = int(figsize[0] / (numvars - 1) * dpi)
                height = int(figsize[1] / (numvars - 1) + 2 * dpi)
        else:
            width = int(figsize[0] / (numvars - 1) * dpi)
            height = int(figsize[1] / (numvars - 1) * dpi)
        return width, height

    if diagonal:
        var = 0
    else:
        var = 1

    if ax is None:
        ax = []
        backend_kwargs.setdefault("width", int(figsize[0] / (numvars - 1) * dpi))
        backend_kwargs.setdefault("height", int(figsize[1] / (numvars - 1) * dpi))
        for row in range(numvars - var):
            row_ax = []
            var1 = (
                flat_var_names[row + var]
                if tmp_flat_var_names is None
                else tmp_flat_var_names[row + var]
            )
            for n, col in enumerate(range(numvars - var)):
                var2 = (
                    flat_var_names[col] if tmp_flat_var_names is None else tmp_flat_var_names[col]
                )
                backend_kwargs_copy = backend_kwargs.copy()
                if "scatter" in kind:
                    tooltips = [
                        (var2, "@{{{}}}".format(var2)),
                        (var1, "@{{{}}}".format(var1)),
                    ]
                    backend_kwargs_copy.setdefault("tooltips", tooltips)
                else:
                    tooltips = None
                if row < col:
                    row_ax.append(None)
                else:
                    jointplot = row == col and numvars == 2 and diagonal
                    rotate = n == 1
                    width, height = get_width_and_height(jointplot, rotate)
                    if jointplot:
                        ax_ = bkp.figure(width=width, height=height, tooltips=tooltips)
                    else:
                        ax_ = bkp.figure(**backend_kwargs_copy)
                    row_ax.append(ax_)
            ax.append(row_ax)
        ax = np.array(ax)
    else:
        assert ax.shape == (numvars - var, numvars - var)
    # pylint: disable=too-many-nested-blocks
    for i in range(0, numvars - var):

        var1 = flat_var_names[i] if tmp_flat_var_names is None else tmp_flat_var_names[i]

        for j in range(0, numvars - var):

            var2 = (
                flat_var_names[j + var]
                if tmp_flat_var_names is None
                else tmp_flat_var_names[j + var]
            )

            if j == i and diagonal:
                rotate = numvars == 2 and j == 1
                var1_dist = infdata_group[i]
                plot_dist(
                    var1_dist,
                    ax=ax[j, i],
                    show=False,
                    backend="bokeh",
                    rotated=rotate,
                    **marginal_kwargs,
                )

                ax[j, i].xaxis.axis_label = flat_var_names[i]
                ax[j, i].yaxis.axis_label = flat_var_names[j + var]

            elif j + var > i:

                if "scatter" in kind:
                    if divergences:
                        ax[j, i].circle(var1, var2, source=source, view=source_nondiv)
                    else:
                        ax[j, i].circle(var1, var2, source=source)

                if "kde" in kind:
                    var1_kde = infdata_group[i]
                    var2_kde = infdata_group[j + var]
                    plot_kde(
                        var1_kde,
                        var2_kde,
                        ax=ax[j, i],
                        backend="bokeh",
                        backend_kwargs={},
                        show=False,
                        **kde_kwargs,
                    )

                if "hexbin" in kind:
                    var1_hexbin = infdata_group[i]
                    var2_hexbin = infdata_group[j + var]
                    ax[j, i].grid.visible = False
                    ax[j, i].hexbin(
                        var1_hexbin, var2_hexbin, **hexbin_kwargs,
                    )

                if divergences:
                    ax[j, i].circle(
                        var1,
                        var2,
                        line_color="black",
                        fill_color="orange",
                        line_width=1,
                        size=10,
                        source=source,
                        view=source_div,
                    )

                if point_estimate:
                    var1_pe = infdata_group[i]
                    var2_pe = infdata_group[j]
                    pe_x = calculate_point_estimate(point_estimate, var1_pe)
                    pe_y = calculate_point_estimate(point_estimate, var2_pe)

                    ax[j, i].square(pe_x, pe_y, line_width=figsize[0] + 1, **point_estimate_kwargs)

                    ax_hline = Span(
                        location=pe_y,
                        dimension="width",
                        line_dash="solid",
                        line_width=3,
                        **point_estimate_kwargs,
                    )
                    ax_vline = Span(
                        location=pe_x,
                        dimension="height",
                        line_dash="solid",
                        line_width=3,
                        **point_estimate_kwargs,
                    )
                    ax[j, i].add_layout(ax_hline)
                    ax[j, i].add_layout(ax_vline)

                    if diagonal:

                        ax[j - 1, i].add_layout(ax_vline)

                        pe_last = calculate_point_estimate(point_estimate, infdata_group[-1])
                        ax_pe_vline = Span(
                            location=pe_last,
                            dimension="height",
                            line_dash="solid",
                            line_width=3,
                            **point_estimate_kwargs,
                        )
                        ax[-1, -1].add_layout(ax_pe_vline)

                        if numvars == 2:
                            ax_pe_hline = Span(
                                location=pe_last,
                                dimension="width",
                                line_dash="solid",
                                line_width=3,
                                **point_estimate_kwargs,
                            )
                            ax[-1, -1].add_layout(ax_pe_hline)

                ax[j, i].xaxis.axis_label = flat_var_names[i]
                ax[j, i].yaxis.axis_label = flat_var_names[j + var]

    show_layout(ax, show)

    return ax
