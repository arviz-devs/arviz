"""Bokeh pairplot."""
import warnings
from copy import deepcopy
from uuid import uuid4

import bokeh.plotting as bkp
import numpy as np
from bokeh.models import CDSView, ColumnDataSource, GroupFilter, Span

from ....rcparams import rcParams
from ...distplot import plot_dist
from ...kdeplot import plot_kde
from ...plot_utils import (
    _scale_fig_size,
    calculate_point_estimate,
    vectorized_to_hex,
    _init_kwargs_dict,
)
from .. import show_layout
from . import backend_kwarg_defaults


def plot_pair(
    ax,
    plotters,
    numvars,
    figsize,
    textsize,
    kind,
    scatter_kwargs,  # pylint: disable=unused-argument
    kde_kwargs,
    hexbin_kwargs,
    gridsize,  # pylint: disable=unused-argument
    colorbar,  # pylint: disable=unused-argument
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
    """Bokeh pair plot."""
    backend_kwargs = _init_kwargs_dict(backend_kwargs)

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }

    hexbin_kwargs = _init_kwargs_dict(hexbin_kwargs)
    hexbin_kwargs.setdefault("size", 0.5)

    marginal_kwargs = _init_kwargs_dict(marginal_kwargs)
    point_estimate_kwargs = _init_kwargs_dict(point_estimate_kwargs)
    kde_kwargs = _init_kwargs_dict(kde_kwargs)

    if kind != "kde":
        kde_kwargs.setdefault("contourf_kwargs", {})
        kde_kwargs["contourf_kwargs"].setdefault("fill_alpha", 0)
        kde_kwargs.setdefault("contour_kwargs", {})
        kde_kwargs["contour_kwargs"].setdefault("line_color", "black")
        kde_kwargs["contour_kwargs"].setdefault("line_alpha", 1)

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

        for dif in difference:
            reference_values_copy[dif] = None

        if difference:
            warn = [dif.replace("\n", " ", 1) for dif in difference]
            warnings.warn(
                "Argument reference_values does not include reference value for: {}".format(
                    ", ".join(warn)
                ),
                UserWarning,
            )

    reference_values_kwargs = _init_kwargs_dict(reference_values_kwargs)
    reference_values_kwargs.setdefault("line_color", "black")
    reference_values_kwargs.setdefault("fill_color", vectorized_to_hex("C2"))
    reference_values_kwargs.setdefault("line_width", 1)
    reference_values_kwargs.setdefault("size", 10)

    divergences_kwargs = _init_kwargs_dict(divergences_kwargs)
    divergences_kwargs.setdefault("line_color", "black")
    divergences_kwargs.setdefault("fill_color", vectorized_to_hex("C1"))
    divergences_kwargs.setdefault("line_width", 1)
    divergences_kwargs.setdefault("size", 10)

    dpi = backend_kwargs.pop("dpi")
    max_plots = (
        numvars**2 if rcParams["plot.max_subplots"] is None else rcParams["plot.max_subplots"]
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

    if numvars == 2:
        offset = 1
    else:
        offset = 2
    (figsize, _, _, _, _, markersize) = _scale_fig_size(
        figsize, textsize, numvars - offset, numvars - offset
    )

    point_estimate_marker_kwargs = _init_kwargs_dict(point_estimate_marker_kwargs)
    point_estimate_marker_kwargs.setdefault("size", markersize)
    point_estimate_marker_kwargs.setdefault("color", "black")
    point_estimate_kwargs.setdefault("line_color", "black")
    point_estimate_kwargs.setdefault("line_width", 2)
    point_estimate_kwargs.setdefault("line_dash", "solid")

    tmp_flat_var_names = None
    if len(flat_var_names) == len(list(set(flat_var_names))):
        source_dict = dict(zip(flat_var_names, [list(post[-1].flatten()) for post in plotters]))
    else:
        tmp_flat_var_names = [f"{name}__{str(uuid4())}" for name in flat_var_names]
        source_dict = dict(zip(tmp_flat_var_names, [list(post[-1].flatten()) for post in plotters]))
    if divergences:
        divergenve_name = f"divergences_{str(uuid4())}"
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

    if marginals:
        marginals_offset = 0
    else:
        marginals_offset = 1

    if ax is None:
        ax = []
        backend_kwargs.setdefault("width", int(figsize[0] / (numvars - 1) * dpi))
        backend_kwargs.setdefault("height", int(figsize[1] / (numvars - 1) * dpi))
        for row in range(numvars - marginals_offset):
            row_ax = []
            var1 = (
                flat_var_names[row + marginals_offset]
                if tmp_flat_var_names is None
                else tmp_flat_var_names[row + marginals_offset]
            )
            for col in range(numvars - marginals_offset):
                var2 = (
                    flat_var_names[col] if tmp_flat_var_names is None else tmp_flat_var_names[col]
                )
                backend_kwargs_copy = backend_kwargs.copy()
                if "scatter" in kind:
                    tooltips = [
                        (var2, f"@{{{var2}}}"),
                        (var1, f"@{{{var1}}}"),
                    ]
                    backend_kwargs_copy.setdefault("tooltips", tooltips)
                else:
                    tooltips = None
                if row < col:
                    row_ax.append(None)
                else:
                    jointplot = row == col and numvars == 2 and marginals
                    rotate = col == 1
                    width, height = get_width_and_height(jointplot, rotate)
                    if jointplot:
                        ax_ = bkp.figure(width=width, height=height, tooltips=tooltips)
                    else:
                        ax_ = bkp.figure(**backend_kwargs_copy)
                    row_ax.append(ax_)
            ax.append(row_ax)
        ax = np.array(ax)
    else:
        assert ax.shape == (numvars - marginals_offset, numvars - marginals_offset)

    # pylint: disable=too-many-nested-blocks
    for i in range(0, numvars - marginals_offset):

        var1 = flat_var_names[i] if tmp_flat_var_names is None else tmp_flat_var_names[i]

        for j in range(0, numvars - marginals_offset):

            var2 = (
                flat_var_names[j + marginals_offset]
                if tmp_flat_var_names is None
                else tmp_flat_var_names[j + marginals_offset]
            )

            if j == i and marginals:
                rotate = numvars == 2 and j == 1
                var1_dist = plotters[i][-1].flatten()
                plot_dist(
                    var1_dist,
                    ax=ax[j, i],
                    show=False,
                    backend="bokeh",
                    rotated=rotate,
                    **marginal_kwargs,
                )

                ax[j, i].xaxis.axis_label = flat_var_names[i]
                ax[j, i].yaxis.axis_label = flat_var_names[j + marginals_offset]

            elif j + marginals_offset > i:

                if "scatter" in kind:
                    if divergences:
                        ax[j, i].circle(var1, var2, source=source, view=source_nondiv)
                    else:
                        ax[j, i].circle(var1, var2, source=source)

                if "kde" in kind:
                    var1_kde = plotters[i][-1].flatten()
                    var2_kde = plotters[j + marginals_offset][-1].flatten()
                    plot_kde(
                        var1_kde,
                        var2_kde,
                        ax=ax[j, i],
                        backend="bokeh",
                        backend_kwargs={},
                        show=False,
                        **deepcopy(kde_kwargs),
                    )

                if "hexbin" in kind:
                    var1_hexbin = plotters[i][-1].flatten()
                    var2_hexbin = plotters[j + marginals_offset][-1].flatten()
                    ax[j, i].grid.visible = False
                    ax[j, i].hexbin(
                        var1_hexbin,
                        var2_hexbin,
                        **hexbin_kwargs,
                    )

                if divergences:
                    ax[j, i].circle(
                        var1,
                        var2,
                        source=source,
                        view=source_div,
                        **divergences_kwargs,
                    )

                if point_estimate:
                    var1_pe = plotters[i][-1].flatten()
                    var2_pe = plotters[j][-1].flatten()
                    pe_x = calculate_point_estimate(point_estimate, var1_pe)
                    pe_y = calculate_point_estimate(point_estimate, var2_pe)
                    ax[j, i].square(pe_x, pe_y, **point_estimate_marker_kwargs)

                    ax_hline = Span(
                        location=pe_y,
                        dimension="width",
                        **point_estimate_kwargs,
                    )
                    ax_vline = Span(
                        location=pe_x,
                        dimension="height",
                        **point_estimate_kwargs,
                    )
                    ax[j, i].add_layout(ax_hline)
                    ax[j, i].add_layout(ax_vline)

                    if marginals:

                        ax[j - 1, i].add_layout(ax_vline)

                        pe_last = calculate_point_estimate(point_estimate, plotters[-1][-1])
                        ax_pe_vline = Span(
                            location=pe_last,
                            dimension="height",
                            **point_estimate_kwargs,
                        )
                        ax[-1, -1].add_layout(ax_pe_vline)

                        if numvars == 2:
                            ax_pe_hline = Span(
                                location=pe_last,
                                dimension="width",
                                **point_estimate_kwargs,
                            )
                            ax[-1, -1].add_layout(ax_pe_hline)

                if reference_values:
                    x = reference_values_copy[flat_var_names[j + marginals_offset]]
                    y = reference_values_copy[flat_var_names[i]]
                    if x and y:
                        ax[j, i].circle(y, x, **reference_values_kwargs)
                ax[j, i].xaxis.axis_label = flat_var_names[i]
                ax[j, i].yaxis.axis_label = flat_var_names[j + marginals_offset]

    show_layout(ax, show)

    return ax
