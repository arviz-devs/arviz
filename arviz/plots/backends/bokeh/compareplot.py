"""Bokeh Compareplot."""

from bokeh.models import Span
from bokeh.models.annotations import Title, Legend


from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_compare(
    ax,
    comp_df,
    legend,
    title,
    figsize,
    plot_ic_diff,
    plot_standard_error,
    insample_dev,
    yticks_pos,
    yticks_labels,
    plot_kwargs,
    textsize,
    information_criterion,
    step,
    backend_kwargs,
    show,
):
    """Bokeh compareplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, _, _, _, line_width, _ = _scale_fig_size(figsize, textsize, 1, 1)

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    yticks_pos = list(yticks_pos)

    labels = []

    if plot_ic_diff:
        ax.yaxis.ticker = yticks_pos[::2]
        ax.yaxis.major_label_overrides = {
            dtype(key): value
            for key, value in zip(yticks_pos, yticks_labels)
            for dtype in (int, float)
            if (dtype(key) - key == 0)
        }

        # create the coordinates for the errorbars
        err_xs = []
        err_ys = []

        for x, y, xerr in zip(
            comp_df[information_criterion].iloc[1:], yticks_pos[1::2], comp_df.dse[1:]
        ):
            err_xs.append((x - xerr, x + xerr))
            err_ys.append((y, y))

        # plot them
        dif_tri = ax.scatter(
            comp_df[information_criterion].iloc[1:],
            yticks_pos[1::2],
            line_color=plot_kwargs.get("color_dse", "grey"),
            fill_color=plot_kwargs.get("color_dse", "grey"),
            line_width=2,
            size=6,
            marker="triangle",
        )
        dif_line = ax.multi_line(err_xs, err_ys, line_color=plot_kwargs.get("color_dse", "grey"))

        labels.append(("ELPD difference", [dif_tri, dif_line]))

    else:
        ax.yaxis.ticker = yticks_pos[::2]
        ax.yaxis.major_label_overrides = dict(zip(yticks_pos[::2], yticks_labels))

    elpd_circ = ax.scatter(
        comp_df[information_criterion],
        yticks_pos[::2],
        line_color=plot_kwargs.get("color_ic", "black"),
        fill_color=None,
        line_width=2,
        size=6,
        marker="circle",
    )
    elpd_label = [elpd_circ]

    if plot_standard_error:
        # create the coordinates for the errorbars
        err_xs = []
        err_ys = []

        for x, y, xerr in zip(comp_df[information_criterion], yticks_pos[::2], comp_df.se):
            err_xs.append((x - xerr, x + xerr))
            err_ys.append((y, y))

        # plot them
        elpd_line = ax.multi_line(err_xs, err_ys, line_color=plot_kwargs.get("color_ic", "black"))
        elpd_label.append(elpd_line)

    labels.append(("ELPD", elpd_label))

    scale = comp_df["scale"].iloc[0]

    if insample_dev:
        p_ic = comp_df[f"p_{information_criterion.split('_')[1]}"]
        if scale == "log":
            correction = p_ic
        elif scale == "negative_log":
            correction = -p_ic
        elif scale == "deviance":
            correction = -(2 * p_ic)
        insample_circ = ax.scatter(
            comp_df[information_criterion] + correction,
            yticks_pos[::2],
            line_color=plot_kwargs.get("color_insample_dev", "black"),
            fill_color=plot_kwargs.get("color_insample_dev", "black"),
            line_width=2,
            size=6,
            marker="circle",
        )
        labels.append(("In-sample ELPD", [insample_circ]))

    vline = Span(
        location=comp_df[information_criterion].iloc[0],
        dimension="height",
        line_color=plot_kwargs.get("color_ls_min_ic", "grey"),
        line_width=line_width,
        line_dash=plot_kwargs.get("ls_min_ic", "dashed"),
    )

    ax.renderers.append(vline)

    if legend:
        legend = Legend(items=labels, orientation="vertical", location="top_right")
        ax.add_layout(legend, "above")
        ax.legend.click_policy = "hide"

    if title:
        _title = Title()
        _title.text = f"Model comparison\n{'higher' if scale == 'log' else 'lower'} is better"
        ax.title = _title

    if scale == "negative_log":
        scale = "-log"

    ax.xaxis.axis_label = f"{information_criterion} ({scale})"
    ax.yaxis.axis_label = "ranked models"
    ax.y_range._property_values["start"] = -1 + step  # pylint: disable=protected-access
    ax.y_range._property_values["end"] = 0 - step  # pylint: disable=protected-access

    show_layout(ax, show)

    return ax
