"""Matplotib Bayesian p-value Posterior predictive plot."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ....stats.density_utils import kde
from ....stats.stats_utils import smooth_data
from ...kdeplot import plot_kde
from ...plot_utils import (
    _scale_fig_size,
    is_valid_quantile,
    sample_reference_distribution,
)
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


def plot_bpv(
    ax,
    length_plotters,
    rows,
    cols,
    obs_plotters,
    pp_plotters,
    total_pp_samples,
    kind,
    t_stat,
    bpv,
    plot_mean,
    reference,
    mse,
    n_ref,
    hdi_prob,
    color,
    figsize,
    textsize,
    labeller,
    plot_ref_kwargs,
    backend_kwargs,
    show,
    smoothing,
):
    """Matplotlib bpv plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, ax_labelsize, _, _, linewidth, markersize = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("squeeze", True)

    if (kind == "u_value") and (reference == "analytical"):
        plot_ref_kwargs = matplotlib_kwarg_dealiaser(plot_ref_kwargs, "fill_between")
    else:
        plot_ref_kwargs = matplotlib_kwarg_dealiaser(plot_ref_kwargs, "plot")

    if kind == "p_value" and reference == "analytical":
        plot_ref_kwargs.setdefault("color", "k")
        plot_ref_kwargs.setdefault("linestyle", "--")
    elif kind == "u_value" and reference == "analytical":
        plot_ref_kwargs.setdefault("color", "k")
        plot_ref_kwargs.setdefault("alpha", 0.2)
    else:
        plot_ref_kwargs.setdefault("alpha", 0.1)
        plot_ref_kwargs.setdefault("color", color)

    if ax is None:
        _, axes = create_axes_grid(length_plotters, rows, cols, backend_kwargs=backend_kwargs)
    else:
        axes = np.asarray(ax)
        if axes.size < length_plotters:
            raise ValueError(
                f"Found {length_plotters} variables to plot but {axes.size} axes instances. "
                "Axes instances must at minimum be equal to variables."
            )

    for i, ax_i in enumerate(np.ravel(axes)[:length_plotters]):
        var_name, selection, isel, obs_vals = obs_plotters[i]
        pp_var_name, _, _, pp_vals = pp_plotters[i]

        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)

        if (obs_vals.dtype.kind == "i" or pp_vals.dtype.kind == "i") and smoothing is True:
            obs_vals, pp_vals = smooth_data(obs_vals, pp_vals)

        if kind == "p_value":
            tstat_pit = np.mean(pp_vals <= obs_vals, axis=-1)
            x_s, tstat_pit_dens = kde(tstat_pit)
            ax_i.plot(x_s, tstat_pit_dens, linewidth=linewidth, color=color)
            ax_i.set_yticks([])
            if reference is not None:
                dist = stats.beta(obs_vals.size / 2, obs_vals.size / 2)
                if reference == "analytical":
                    lwb = dist.ppf((1 - 0.9999) / 2)
                    upb = 1 - lwb
                    x = np.linspace(lwb, upb, 500)
                    dens_ref = dist.pdf(x)
                    ax_i.plot(x, dens_ref, zorder=1, **plot_ref_kwargs)
                elif reference == "samples":
                    x_ss, u_dens = sample_reference_distribution(
                        dist,
                        (
                            tstat_pit_dens.size,
                            n_ref,
                        ),
                    )
                    ax_i.plot(x_ss, u_dens, linewidth=linewidth, **plot_ref_kwargs)

        elif kind == "u_value":
            tstat_pit = np.mean(pp_vals <= obs_vals, axis=0)
            x_s, tstat_pit_dens = kde(tstat_pit)
            ax_i.plot(x_s, tstat_pit_dens, color=color)
            if reference is not None:
                if reference == "analytical":
                    n_obs = obs_vals.size
                    hdi_ = stats.beta(n_obs / 2, n_obs / 2).ppf((1 - hdi_prob) / 2)
                    hdi_odds = (hdi_ / (1 - hdi_), (1 - hdi_) / hdi_)
                    ax_i.axhspan(*hdi_odds, **plot_ref_kwargs)
                    ax_i.axhline(1, color="w", zorder=1)
                elif reference == "samples":
                    dist = stats.uniform(0, 1)
                    x_ss, u_dens = sample_reference_distribution(dist, (tstat_pit_dens.size, n_ref))
                    ax_i.plot(x_ss, u_dens, linewidth=linewidth, **plot_ref_kwargs)
            if mse:
                ax_i.plot(0, 0, label=f"mse={np.mean((1 - tstat_pit_dens)**2) * 100:.2f}")
                ax_i.legend()

            ax_i.set_ylim(0, None)
            ax_i.set_xlim(0, 1)
        else:
            if t_stat in ["mean", "median", "std"]:
                if t_stat == "mean":
                    tfunc = np.mean
                elif t_stat == "median":
                    tfunc = np.median
                elif t_stat == "std":
                    tfunc = np.std
                obs_vals = tfunc(obs_vals)
                pp_vals = tfunc(pp_vals, axis=1)
            elif hasattr(t_stat, "__call__"):
                obs_vals = t_stat(obs_vals.flatten())
                pp_vals = t_stat(pp_vals)
            elif is_valid_quantile(t_stat):
                t_stat = float(t_stat)
                obs_vals = np.quantile(obs_vals, q=t_stat)
                pp_vals = np.quantile(pp_vals, q=t_stat, axis=1)
            else:
                raise ValueError(f"T statistics {t_stat} not implemented")

            plot_kde(pp_vals, ax=ax_i, plot_kwargs={"color": color})
            ax_i.set_yticks([])
            if bpv:
                p_value = np.mean(pp_vals <= obs_vals)
                ax_i.plot(obs_vals, 0, label=f"bpv={p_value:.2f}", alpha=0)
                ax_i.legend()

            if plot_mean:
                ax_i.plot(
                    obs_vals.mean(), 0, "o", color=color, markeredgecolor="k", markersize=markersize
                )

        ax_i.set_title(
            labeller.make_pp_label(var_name, pp_var_name, selection, isel), fontsize=ax_labelsize
        )

    if backend_show(show):
        plt.show()

    return axes
