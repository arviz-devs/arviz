"""Matplotlib Parallel coordinates plot."""
import matplotlib.pyplot as plt
import numpy as np


def plot_parallel(
    ax,
    colornd,
    colord,
    shadend,
    diverging_mask,
    _posterior,
    textsize,
    var_names,
    xt_labelsize,
    legend,
    figsize,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(_posterior[:, ~diverging_mask], color=colornd, alpha=shadend)

    if np.any(diverging_mask):
        ax.plot(_posterior[:, diverging_mask], color=colord, lw=1)

    ax.tick_params(labelsize=textsize)
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names)

    if legend:
        ax.plot([], color=colornd, label="non-divergent")
        if np.any(diverging_mask):
            ax.plot([], color=colord, label="divergent")
        ax.legend(fontsize=xt_labelsize)

    return ax
