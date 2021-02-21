"""Bokeh Compareplot."""
from bokeh.models import Span

from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_compare(
    ax,
    comp_df,
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

    if plot_ic_diff:
        ax.yaxis.ticker = yticks_pos
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
        ax.triangle(
            comp_df[information_criterion].iloc[1:],
            yticks_pos[1::2],
            line_color=plot_kwargs.get("color_dse", "grey"),
            fill_color=plot_kwargs.get("color_dse", "grey"),
            line_width=2,
            size=6,
        )
        ax.multi_line(err_xs, err_ys, line_color=plot_kwargs.get("color_dse", "grey"))

    else:
        ax.yaxis.ticker = yticks_pos[::2]
        ax.yaxis.major_label_overrides = {
            key: value for key, value in zip(yticks_pos[::2], yticks_labels)
        }

    ax.circle(
        comp_df[information_criterion],
        yticks_pos[::2],
        line_color=plot_kwargs.get("color_ic", "black"),
        fill_color=None,
        line_width=2,
        size=6,
    )

    if plot_standard_error:
        # create the coordinates for the errorbars
        err_xs = []
        err_ys = []

        for x, y, xerr in zip(comp_df[information_criterion], yticks_pos[::2], comp_df.se):
            err_xs.append((x - xerr, x + xerr))
            err_ys.append((y, y))

        # plot them
        ax.multi_line(err_xs, err_ys, line_color=plot_kwargs.get("color_ic", "black"))

    if insample_dev:
        scale = comp_df[f"{information_criterion}_scale"][0]
        p_ic = comp_df[f"p_{information_criterion}"]
        if scale == "log":
            correction = p_ic
        elif scale == "negative_log":
            correction = -p_ic
        elif scale == "deviance":
            correction = -(2 * p_ic)
        ax.circle(
            comp_df[information_criterion] + correction,
            yticks_pos[::2],
            line_color=plot_kwargs.get("color_insample_dev", "black"),
            fill_color=plot_kwargs.get("color_insample_dev", "black"),
            line_width=2,
            size=6,
        )

    vline = Span(
        location=comp_df[information_criterion].iloc[0],
        dimension="height",
        line_color=plot_kwargs.get("color_ls_min_ic", "grey"),
        line_width=line_width,
        line_dash=plot_kwargs.get("ls_min_ic", "dashed"),
    )

    ax.renderers.append(vline)

    scale_col = information_criterion + "_scale"
    if scale_col in comp_df:
        scale = comp_df[scale_col].iloc[0].capitalize()
    else:
        scale = "Deviance"
    ax.xaxis.axis_label = scale
    ax.y_range._property_values["start"] = -1 + step  # pylint: disable=protected-access
    ax.y_range._property_values["end"] = 0 - step  # pylint: disable=protected-access

    show_layout(ax, show)

    return ax
