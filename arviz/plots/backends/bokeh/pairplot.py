"""Bokeh pairplot."""
import warnings
from uuid import uuid4

import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
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
    plot_kwargs,
    contour,
    fill_last,
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

    if marginal_kwargs is None:
        marginal_kwargs = {}

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

    (figsize, _, _, _, linewidth, _) = _scale_fig_size(figsize, textsize, numvars - 2, numvars - 2)

    if ax is None:
        ax = []
        backend_kwargs.setdefault("width", int(figsize[0] / (numvars - 1) * dpi))
        backend_kwargs.setdefault("height", int(figsize[1] / (numvars - 1) * dpi))
        if diagonal:
            for row in range(numvars):
                row_ax = []            
                for col in range(numvars):
                    if row < col:
                        row_ax.append(None)
                    elif row == col and numvars == 2:
                        ax_ = bkp.figure(
                                width=int(figsize[0] / (numvars - 1) * dpi),
                                height=int(figsize[1] / (numvars - 1) + 2 * dpi),
                            )
                        row_ax.append(ax_)
                    else:
                        ax_ = bkp.figure(
                                **backend_kwargs
                            )
                        row_ax.append(ax_)
                ax.append(row_ax)
            ax = np.array(ax)
        else:
            for row in range(numvars - 1):
                row_ax = []
                for col in range(numvars - 1):
                    if row < col:
                        row_ax.append(None)
                    else:
                        ax_ = bkp.figure(**backend_kwargs)
                        row_ax.append(ax_)

                ax.append(row_ax)
            ax = np.array(ax)
    
    tmp_flat_var_names = None
    if len(flat_var_names) == len(list(set(flat_var_names))):
        source_dict = dict(zip(flat_var_names, [list(post) for post in infdata_group]))
    else:
        tmp_flat_var_names = ["{}__{}".format(name, str(uuid4())) for name in flat_var_names]
        source_dict = dict(zip(tmp_flat_var_names, [list(post) for post in infdata_group]))
    if divergences:
        divergenve_name = "divergences_{}".format(str(uuid4()))
        source_dict[divergenve_name] = (
            np.array(diverging_mask).astype(bool).astype(int).astype(str)
        )

    source = ColumnDataSource(data=source_dict)

    if divergences:
        source_nondiv = CDSView(
            source=source, filters=[GroupFilter(column_name=divergenve_name, group="0")]
        )
        source_div = CDSView(
            source=source, filters=[GroupFilter(column_name=divergenve_name, group="1")]
        )
    
    if diagonal:
        var = 0
    else:
        var = 1
    
    for i in range(0, numvars - var):
        var1 = flat_var_names[i] if tmp_flat_var_names is None else tmp_flat_var_names[i]

        for j in range(0, numvars - var):

            var2 = flat_var_names[j + var] if tmp_flat_var_names is None else tmp_flat_var_names[j + var]

            if j == i and diagonal:

                var1_dist = infdata_group[i]
                plot_dist(var1_dist, ax=ax[j, i], show=False, backend="bokeh", **marginal_kwargs)
            
            if j + var > i:
                
                if kind == "scatter":
                    if divergences:
                        ax[j, i].circle(var1, var2, source=source, view=source_nondiv)
                    else:
                        ax[j, i].circle(var1, var2, source=source)

                elif kind == "kde":
                    var1_kde = infdata_group[i]
                    var2_kde = infdata_group[j+ var]
                    plot_kde(
                        var1_kde,
                        var2_kde,
                        contour=contour,
                        fill_last=fill_last,
                        ax=ax[j, i],
                        backend="bokeh",
                        backend_kwargs={},
                        show=False,
                        **plot_kwargs
                    )

                elif kind == "hexbin":
                    var1_hexbin = infdata_group[i]
                    var2_hexbin = infdata_group[j + var]
                    ax[j, i].grid.visible = False
                    ax[j, i].hexbin(var1_hexbin, var2_hexbin, size=0.5)
                else:
                    
                    ax[j, i].circle(flat_var_names[0], flat_var_names[1], source=source)
                    plot_kde(
                        infdata_group[0],
                        infdata_group[1],
                        ax=ax[j, i],
                        fill_last=False,
                        backend="bokeh",
                        show=False,
                        contour_kwargs={"fill_alpha": 0, "line_alpha": 1},
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
                    pe_x = calculate_point_estimate(point_estimate, var1_pe, 0.5)
                    pe_y = calculate_point_estimate(point_estimate, var2_pe, 0.5)

                    ax[j, i].square(
                        pe_x,
                        pe_y,
                        line_width=figsize[0] + 4,
                        line_color="red",
                        **point_estimate_kwargs
                    )

                    ax_hline = Span(
                        location=pe_y,
                        dimension="width",
                        line_color="red",
                        line_dash="solid",
                        line_width=3,
                    )
                    ax_vline = Span(
                        location=pe_x,
                        dimension="height",
                        line_color="red",
                        line_dash="solid",
                        line_width=3,
                    )
                    ax[j, i].add_layout(ax_hline)
                    ax[j, i].add_layout(ax_vline)

                    if diagonal:

                        ax[j - 1, i].add_layout(ax_vline)

                        pe = calculate_point_estimate(point_estimate, infdata_group[-1], 0.5)
                        ax_pe_vline = Span(
                            location=pe,
                            dimension="height",
                            line_color="red",
                            line_dash="solid",
                            line_width=3,
                        )
                        ax[-1, -1].add_layout(ax_pe_vline)

                ax[j, i].xaxis.axis_label = flat_var_names[i]
                ax[j, i].yaxis.axis_label = flat_var_names[j + var]

    show_layout(ax, show)

    return ax
