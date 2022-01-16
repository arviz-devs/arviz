"""Matplotlib energyplot."""
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import rcParams as mpl_rcParams

from ....stats import bfmi as e_bfmi
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


def plot_energy(
    ax,
    energy,
    kind,
    bfmi,
    figsize,
    textsize,
    fill_alpha,
    fill_color,
    fill_kwargs,
    plot_kwargs,
    bw,
    legend,
    backend_kwargs,
    show,
):
    """Matplotlib energy plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, _, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, textsize, 1, 1)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True
    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    fill_kwargs = matplotlib_kwarg_dealiaser(fill_kwargs, "hexbin")
    types = "hist" if kind == "hist" else "plot"
    plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, types)

    _colors = [
        prop for _, prop in zip(range(10), cycle(mpl_rcParams["axes.prop_cycle"].by_key()["color"]))
    ]
    if (fill_color[0].startswith("C") and len(fill_color[0]) == 2) and (
        fill_color[1].startswith("C") and len(fill_color[1]) == 2
    ):
        fill_color = tuple((_colors[int(color[1:]) % 10] for color in fill_color))
    elif fill_color[0].startswith("C") and len(fill_color[0]) == 2:
        fill_color = tuple([_colors[int(fill_color[0][1:]) % 10]] + list(fill_color[1:]))
    elif fill_color[1].startswith("C") and len(fill_color[1]) == 2:
        fill_color = tuple(list(fill_color[1:]) + [_colors[int(fill_color[0][1:]) % 10]])

    series = zip(
        fill_alpha,
        fill_color,
        ("Marginal Energy", "Energy transition"),
        (energy - energy.mean(), np.diff(energy)),
    )

    if kind == "kde":
        for alpha, color, label, value in series:
            fill_kwargs["alpha"] = alpha
            fill_kwargs["color"] = color
            plot_kwargs.setdefault("color", color)
            plot_kwargs.setdefault("alpha", 0)
            plot_kwargs.setdefault("linewidth", linewidth)
            plot_kde(
                value,
                bw=bw,
                label=label,
                textsize=xt_labelsize,
                fill_kwargs=fill_kwargs,
                plot_kwargs=plot_kwargs,
                ax=ax,
                legend=False,
            )
    elif kind == "hist":
        for alpha, color, label, value in series:
            ax.hist(
                value.flatten(),
                bins="auto",
                density=True,
                alpha=alpha,
                label=label,
                color=color,
                **plot_kwargs,
            )

    else:
        raise ValueError(f"Plot type {kind} not recognized.")

    if bfmi:
        for idx, val in enumerate(e_bfmi(energy)):
            ax.plot([], label=f"chain {idx:>2} BFMI = {val:.2f}", alpha=0)
    if legend:
        ax.legend()

    ax.set_xticks([])
    ax.set_yticks([])

    if backend_show(show):
        plt.show()

    return ax
