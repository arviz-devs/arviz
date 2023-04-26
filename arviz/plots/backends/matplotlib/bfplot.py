import matplotlib.pyplot as plt

from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size


def plot_bf(
    ax,
    bf_10,
    bf_01,
    prior,
    posterior,
    ref_val,
    prior_at_ref_val,
    posterior_at_ref_val,
    var_name,
    colors,
    figsize,
    textsize,
    plot_kwargs,
    hist_kwargs,
    backend_kwargs,
    show,
):

    """Matplotlib Bayes Factor plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    if hist_kwargs is None:
        hist_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, _, _, _, linewidth, _ = _scale_fig_size(figsize, textsize, 1, 1)

    plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "plot")
    plot_kwargs.setdefault("linewidth", linewidth)
    hist_kwargs.setdefault("alpha", 0.5)

    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("squeeze", True)

    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    plot_dist(
        prior,
        color=colors[0],
        label="Prior",
        ax=ax,
        plot_kwargs=plot_kwargs,
        hist_kwargs=hist_kwargs,
    )
    plot_dist(
        posterior,
        color=colors[1],
        label="Posterior",
        ax=ax,
        plot_kwargs=plot_kwargs,
        hist_kwargs=hist_kwargs,
    )

    ax.plot(ref_val, posterior_at_ref_val, "ko", lw=1.5)
    ax.plot(ref_val, prior_at_ref_val, "ko", lw=1.5)
    ax.axvline(ref_val, color="k", ls="--")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Density")
    ax.set_title(f"The BF_10 is {bf_10:.2f}\nThe BF_01 is {bf_01:.2f}")
    plt.legend()

    if backend_show(show):
        plt.show()

    return ax
