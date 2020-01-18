"""Matplotlib energyplot."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from . import backend_show
from ...plot_utils import (
    make_label,
    _create_axes_grid,
)


def plot_ess(
    ax,
    plotters,
    xdata,
    ess_tail_dataset,
    mean_ess,
    sd_ess,
    idata,
    data,
    text_x,
    text_va,
    kind,
    extra_methods,
    rows,
    cols,
    figsize,
    kwargs,
    extra_kwargs,
    text_kwargs,
    _linewidth,
    _markersize,
    n_samples,
    relative,
    min_ess,
    xt_labelsize,
    titlesize,
    ax_labelsize,
    ylabel,
    rug,
    rug_kind,
    rug_kwargs,
    hline_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib ess plot."""
    if ax is None:
        _, ax = _create_axes_grid(
            len(plotters),
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
            backend_kwargs=backend_kwargs,
        )

    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
        ax_.plot(xdata, x, **kwargs)
        if kind == "evolution":
            ess_tail = ess_tail_dataset[var_name].sel(**selection)
            ax_.plot(xdata, ess_tail, **extra_kwargs)
        elif rug:
            if rug_kwargs is None:
                rug_kwargs = {}
            if not hasattr(idata, "sample_stats"):
                raise ValueError("InferenceData object must contain sample_stats for rug plot")
            if not hasattr(idata.sample_stats, rug_kind):
                raise ValueError("InferenceData does not contain {} data".format(rug_kind))
            rug_kwargs.setdefault("marker", "|")
            rug_kwargs.setdefault("linestyle", rug_kwargs.pop("ls", "None"))
            rug_kwargs.setdefault("color", rug_kwargs.pop("c", kwargs.get("color", "C0")))
            rug_kwargs.setdefault("space", 0.1)
            rug_kwargs.setdefault("markersize", rug_kwargs.pop("ms", 2 * _markersize))

            values = data[var_name].sel(**selection).values.flatten()
            mask = idata.sample_stats[rug_kind].values.flatten()
            values = rankdata(values)[mask]
            rug_space = np.max(x) * rug_kwargs.pop("space")
            rug_x, rug_y = values / (len(mask) - 1), np.zeros_like(values) - rug_space
            ax_.plot(rug_x, rug_y, **rug_kwargs)
            ax_.axhline(0, color="k", linewidth=_linewidth, alpha=0.7)
        if extra_methods:
            mean_ess_i = mean_ess[var_name].sel(**selection).values.item()
            sd_ess_i = sd_ess[var_name].sel(**selection).values.item()
            ax_.axhline(mean_ess_i, **extra_kwargs)
            ax_.annotate(
                "mean",
                (text_x, mean_ess_i),
                va=text_va
                if text_va is not None
                else "bottom"
                if mean_ess_i >= sd_ess_i
                else "top",
                **text_kwargs,
            )
            ax_.axhline(sd_ess_i, **extra_kwargs)
            ax_.annotate(
                "sd",
                (text_x, sd_ess_i),
                va=text_va if text_va is not None else "bottom" if sd_ess_i > mean_ess_i else "top",
                **text_kwargs,
            )

        ax_.axhline(400 / n_samples if relative else min_ess, **hline_kwargs)

        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.set_xlabel(
            "Total number of draws" if kind == "evolution" else "Quantile", fontsize=ax_labelsize
        )
        ax_.set_ylabel(
            ylabel.format("Relative ESS" if relative else "ESS"), fontsize=ax_labelsize, wrap=True
        )
        if kind == "evolution":
            ax_.legend(title="Method", fontsize=xt_labelsize, title_fontsize=xt_labelsize)
        else:
            ax_.set_xlim(0, 1)
        if rug:
            ax_.yaxis.get_major_locator().set_params(nbins="auto", steps=[1, 2, 5, 10])
            _, ymax = ax_.get_ylim()
            yticks = ax_.get_yticks().astype(np.int64)
            yticks = yticks[(yticks >= 0) & (yticks < ymax)]
            ax_.set_yticks(yticks)
            ax_.set_yticklabels(yticks)
        else:
            ax_.set_ylim(bottom=0)

    if backend_show(show):
        plt.show()

    return ax
