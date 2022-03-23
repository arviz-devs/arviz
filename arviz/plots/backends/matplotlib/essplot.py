"""Matplotlib energyplot."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


def plot_ess(
    ax,
    plotters,
    xdata,
    ess_tail_dataset,
    mean_ess,
    sd_ess,
    idata,
    data,
    kind,
    extra_methods,
    textsize,
    rows,
    cols,
    figsize,
    kwargs,
    extra_kwargs,
    text_kwargs,
    n_samples,
    relative,
    min_ess,
    labeller,
    ylabel,
    rug,
    rug_kind,
    rug_kwargs,
    hline_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib ess plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    kwargs = matplotlib_kwarg_dealiaser(kwargs, "plot")
    _linestyle = "-" if kind == "evolution" else "none"
    kwargs.setdefault("linestyle", _linestyle)
    kwargs.setdefault("linewidth", _linewidth)
    kwargs.setdefault("markersize", _markersize)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("zorder", 3)

    extra_kwargs = matplotlib_kwarg_dealiaser(extra_kwargs, "plot")
    if kind == "evolution":
        extra_kwargs = {
            **extra_kwargs,
            **{key: item for key, item in kwargs.items() if key not in extra_kwargs},
        }
        kwargs.setdefault("label", "bulk")
        extra_kwargs.setdefault("label", "tail")
    else:
        extra_kwargs.setdefault("linewidth", _linewidth / 2)
        extra_kwargs.setdefault("color", "k")
        extra_kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("label", kind)

    hline_kwargs = matplotlib_kwarg_dealiaser(hline_kwargs, "plot")
    hline_kwargs.setdefault("linewidth", _linewidth)
    hline_kwargs.setdefault("linestyle", "--")
    hline_kwargs.setdefault("color", "gray")
    hline_kwargs.setdefault("alpha", 0.7)
    if extra_methods:
        text_kwargs = matplotlib_kwarg_dealiaser(text_kwargs, "text")
        text_x = text_kwargs.pop("x", 1)
        text_kwargs.setdefault("fontsize", xt_labelsize * 0.7)
        text_kwargs.setdefault("alpha", extra_kwargs["alpha"])
        text_kwargs.setdefault("color", extra_kwargs["color"])
        text_kwargs.setdefault("horizontalalignment", "right")
        text_va = text_kwargs.pop("verticalalignment", None)

    if ax is None:
        _, ax = create_axes_grid(
            len(plotters),
            rows,
            cols,
            backend_kwargs=backend_kwargs,
        )

    for (var_name, selection, isel, x), ax_ in zip(plotters, np.ravel(ax)):
        ax_.plot(xdata, x, **kwargs)
        if kind == "evolution":
            ess_tail = ess_tail_dataset[var_name].sel(**selection)
            ax_.plot(xdata, ess_tail, **extra_kwargs)
        elif rug:
            rug_kwargs = matplotlib_kwarg_dealiaser(rug_kwargs, "plot")
            if not hasattr(idata, "sample_stats"):
                raise ValueError("InferenceData object must contain sample_stats for rug plot")
            if not hasattr(idata.sample_stats, rug_kind):
                raise ValueError(f"InferenceData does not contain {rug_kind} data")
            rug_kwargs.setdefault("marker", "|")
            rug_kwargs.setdefault("linestyle", rug_kwargs.pop("ls", "None"))
            rug_kwargs.setdefault("color", rug_kwargs.pop("c", kwargs.get("color", "C0")))
            rug_kwargs.setdefault("space", 0.1)
            rug_kwargs.setdefault("markersize", rug_kwargs.pop("ms", 2 * _markersize))

            values = data[var_name].sel(**selection).values.flatten()
            mask = idata.sample_stats[rug_kind].values.flatten()
            values = rankdata(values, method="average")[mask]
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

        if relative and kind == "evolution":
            thin_xdata = np.linspace(xdata.min(), xdata.max(), 100)
            ax_.plot(thin_xdata, min_ess / thin_xdata, **hline_kwargs)
        else:
            hline = min_ess / n_samples if relative else min_ess
            ax_.axhline(hline, **hline_kwargs)

        ax_.set_title(
            labeller.make_label_vert(var_name, selection, isel), fontsize=titlesize, wrap=True
        )
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
