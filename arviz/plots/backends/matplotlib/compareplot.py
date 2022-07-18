"""Matplotlib Compareplot."""
import matplotlib.pyplot as plt

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid


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
    information_criterion,
    textsize,
    step,
    backend_kwargs,
    show,
):
    """Matplotlib compare plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if figsize is None:
        figsize = (6, len(comp_df))

    figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, textsize, 1, 1)

    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    if plot_standard_error:
        ax.errorbar(
            x=comp_df[information_criterion],
            y=yticks_pos[::2],
            xerr=comp_df.se,
            label="ELPD",
            color=plot_kwargs.get("color_ic", "k"),
            fmt=plot_kwargs.get("marker_ic", "o"),
            mfc=plot_kwargs.get("marker_fc", "white"),
            mew=linewidth,
            lw=linewidth,
        )
    else:
        ax.plot(
            comp_df[information_criterion],
            yticks_pos[::2],
            label="ELPD",
            color=plot_kwargs.get("color_ic", "k"),
            marker=plot_kwargs.get("marker_ic", "o"),
            mfc=plot_kwargs.get("marker_fc", "white"),
            mew=linewidth,
            lw=0,
            zorder=3,
        )

    if plot_ic_diff:
        ax.set_yticks(yticks_pos)
        ax.errorbar(
            x=comp_df[information_criterion].iloc[1:],
            y=yticks_pos[1::2],
            xerr=comp_df.dse[1:],
            label="ELPD difference",
            color=plot_kwargs.get("color_dse", "grey"),
            fmt=plot_kwargs.get("marker_dse", "^"),
            mew=linewidth,
            elinewidth=linewidth,
        )

    else:
        ax.set_yticks(yticks_pos[::2])

    scale = comp_df["scale"][0]

    if insample_dev:
        p_ic = comp_df[f"p_{information_criterion.split('_')[1]}"]
        if scale == "log":
            correction = p_ic
        elif scale == "negative_log":
            correction = -p_ic
        elif scale == "deviance":
            correction = -(2 * p_ic)
        ax.plot(
            comp_df[information_criterion] + correction,
            yticks_pos[::2],
            label="In-sample ELPD",
            color=plot_kwargs.get("color_insample_dev", "k"),
            marker=plot_kwargs.get("marker_insample_dev", "o"),
            mew=linewidth,
            lw=0,
        )

    ax.axvline(
        comp_df[information_criterion].iloc[0],
        ls=plot_kwargs.get("ls_min_ic", "--"),
        color=plot_kwargs.get("color_ls_min_ic", "grey"),
        lw=linewidth,
    )
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1, fontsize=ax_labelsize)

    if title:
        ax.set_title(
            f"Model comparison\n{'higher' if scale == 'log' else 'lower'} is better",
            fontsize=ax_labelsize,
        )

    if scale == "negative_log":
        scale = "-log"

    ax.set_xlabel(f"{information_criterion} ({scale})", fontsize=ax_labelsize)
    ax.set_ylabel("ranked models", fontsize=ax_labelsize)
    ax.set_yticklabels(yticks_labels)
    ax.set_ylim(-1 + step, 0 - step)
    ax.tick_params(labelsize=xt_labelsize)

    if backend_show(show):
        plt.show()

    return ax
