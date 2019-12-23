"""Matplotlib Compareplot."""
import matplotlib.pyplot as plt

from . import backend_kwarg_defaults, backend_show


def plot_compare(
    ax,
    comp_df,
    figsize,
    plot_ic_diff,
    plot_standard_error,
    insample_dev,
    yticks_pos,
    yticks_labels,
    linewidth,
    plot_kwargs,
    information_criterion,
    ax_labelsize,
    xt_labelsize,
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
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, **backend_kwargs)

    if plot_ic_diff:
        yticks_labels[0] = comp_df.index[0]
        yticks_labels[2::2] = comp_df.index[1:]
        ax.set_yticks(yticks_pos)
        ax.errorbar(
            x=comp_df[information_criterion].iloc[1:],
            y=yticks_pos[1::2],
            xerr=comp_df.dse[1:],
            color=plot_kwargs.get("color_dse", "grey"),
            fmt=plot_kwargs.get("marker_dse", "^"),
            mew=linewidth,
            elinewidth=linewidth,
        )

    else:
        yticks_labels = comp_df.index
        ax.set_yticks(yticks_pos[::2])

    if plot_standard_error:
        ax.errorbar(
            x=comp_df[information_criterion],
            y=yticks_pos[::2],
            xerr=comp_df.se,
            color=plot_kwargs.get("color_ic", "k"),
            fmt=plot_kwargs.get("marker_ic", "o"),
            mfc="None",
            mew=linewidth,
            lw=linewidth,
        )
    else:
        ax.plot(
            comp_df[information_criterion],
            yticks_pos[::2],
            color=plot_kwargs.get("color_ic", "k"),
            marker=plot_kwargs.get("marker_ic", "o"),
            mfc="None",
            mew=linewidth,
            lw=0,
        )

    if insample_dev:
        ax.plot(
            comp_df[information_criterion] - (2 * comp_df["p_" + information_criterion]),
            yticks_pos[::2],
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

    scale_col = information_criterion + "_scale"
    if scale_col in comp_df:
        scale = comp_df[scale_col].iloc[0].capitalize()
    else:
        scale = "Deviance"
    ax.set_xlabel(scale, fontsize=ax_labelsize)
    ax.set_yticklabels(yticks_labels)
    ax.set_ylim(-1 + step, 0 - step)
    ax.tick_params(labelsize=xt_labelsize)

    if backend_show(show):
        plt.show()

    return ax
