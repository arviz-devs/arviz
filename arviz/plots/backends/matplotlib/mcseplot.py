"""Matplotlib mcseplot."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from . import backend_show
from ....stats.stats_utils import quantile as _quantile
from ...plot_utils import (
    make_label,
    _create_axes_grid,
)


def plot_mcse(
    ax,
    plotters,
    length_plotters,
    rows,
    cols,
    figsize,
    errorbar,
    rug,
    data,
    probs,
    kwargs,
    extra_methods,
    mean_mcse,
    sd_mcse,
    text_x,
    text_va,
    text_kwargs,
    rug_kwargs,
    extra_kwargs,
    idata,
    rug_kind,
    _markersize,
    _linewidth,
    xt_labelsize,
    ax_labelsize,
    titlesize,
    backend_kwargs,
    show,
):
    """Matplotlib mcseplot."""
    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
            backend_kwargs=backend_kwargs,
        )

    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
        if errorbar or rug:
            values = data[var_name].sel(**selection).values.flatten()
        if errorbar:
            quantile_values = _quantile(values, probs)
            ax_.errorbar(probs, quantile_values, yerr=x, **kwargs)
        else:
            ax_.plot(probs, x, label="quantile", **kwargs)
            if extra_methods:
                mean_mcse_i = mean_mcse[var_name].sel(**selection).values.item()
                sd_mcse_i = sd_mcse[var_name].sel(**selection).values.item()
                ax_.axhline(mean_mcse_i, **extra_kwargs)
                ax_.annotate(
                    "mean",
                    (text_x, mean_mcse_i),
                    va=text_va
                    if text_va is not None
                    else "bottom"
                    if mean_mcse_i > sd_mcse_i
                    else "top",
                    **text_kwargs,
                )
                ax_.axhline(sd_mcse_i, **extra_kwargs)
                ax_.annotate(
                    "sd",
                    (text_x, sd_mcse_i),
                    va=text_va
                    if text_va is not None
                    else "bottom"
                    if sd_mcse_i >= mean_mcse_i
                    else "top",
                    **text_kwargs,
                )
        if rug:
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

            mask = idata.sample_stats[rug_kind].values.flatten()
            values = rankdata(values)[mask]
            y_min, y_max = ax_.get_ylim()
            y_min = y_min if errorbar else 0
            rug_space = (y_max - y_min) * rug_kwargs.pop("space")
            rug_x, rug_y = values / (len(mask) - 1), np.full_like(values, y_min) - rug_space
            ax_.plot(rug_x, rug_y, **rug_kwargs)
            ax_.axhline(y_min, color="k", linewidth=_linewidth, alpha=0.7)

        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.set_xlabel("Quantile", fontsize=ax_labelsize, wrap=True)
        ax_.set_ylabel(
            r"Value $\pm$ MCSE for quantiles" if errorbar else "MCSE for quantiles",
            fontsize=ax_labelsize,
            wrap=True,
        )
        ax_.set_xlim(0, 1)
        if rug:
            ax_.yaxis.get_major_locator().set_params(nbins="auto", steps=[1, 2, 5, 10])
            y_min, y_max = ax_.get_ylim()
            yticks = ax_.get_yticks()
            yticks = yticks[(yticks >= y_min) & (yticks < y_max)]
            ax_.set_yticks(yticks)
            ax_.set_yticklabels(["{:.3g}".format(ytick) for ytick in yticks])
        elif not errorbar:
            ax_.set_ylim(bottom=0)

    if backend_show(show):
        plt.show()

    return ax
