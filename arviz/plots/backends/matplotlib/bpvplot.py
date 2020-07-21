"""Matplotib Bayesian p-value Posterior predictive plot."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from . import backend_show
from ...kdeplot import plot_kde
from ...plot_utils import (
    make_label,
    _create_axes_grid,
    _scale_fig_size,
    sample_reference_distribution,
    is_valid_quantile,
)
from ....numeric_utils import _fast_kde


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
    n_ref,
    hdi_prob,
    color,
    figsize,
    textsize,
    plot_ref_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib bpv plot."""
    figsize, ax_labelsize, _, _, linewidth, markersize = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    if plot_ref_kwargs is None:
        plot_ref_kwargs = {}
    if kind == "p_value" and reference == "analytical":
        plot_ref_kwargs.setdefault("color", "k")
        plot_ref_kwargs.setdefault("linestyle", "--")
    else:
        plot_ref_kwargs.setdefault("alpha", 0.1)
        plot_ref_kwargs.setdefault("color", color)

    if ax is None:
        _, axes = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, backend_kwargs=backend_kwargs
        )
    else:
        axes = np.ravel(ax)
        if len(axes) != length_plotters:
            raise ValueError(
                "Found {} variables to plot but {} axes instances. They must be equal.".format(
                    length_plotters, len(axes)
                )
            )

    for i, ax_i in enumerate(axes):
        var_name, selection, obs_vals = obs_plotters[i]
        pp_var_name, _, pp_vals = pp_plotters[i]

        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)

        if kind == "p_value":
            tstat_pit = np.mean(pp_vals <= obs_vals, axis=-1)
            tstat_pit_dens, xmin, xmax = _fast_kde(tstat_pit)
            x_s = np.linspace(xmin, xmax, len(tstat_pit_dens))
            ax_i.plot(x_s, tstat_pit_dens, linewidth=linewidth, color=color)
            ax_i.set_yticks([])
            if reference is not None:
                dist = stats.beta(obs_vals.size / 2, obs_vals.size / 2)
                if reference == "analytical":
                    lwb = dist.ppf((1 - 0.9999) / 2)
                    upb = 1 - lwb
                    x = np.linspace(lwb, upb, 500)
                    dens_ref = dist.pdf(x)
                    ax_i.plot(x, dens_ref, **plot_ref_kwargs)
                elif reference == "samples":
                    x_ss, u_dens = sample_reference_distribution(
                        dist, (n_ref, tstat_pit_dens.size,)
                    )
                    ax_i.plot(x_ss, u_dens, linewidth=linewidth, **plot_ref_kwargs)

        elif kind == "u_value":
            tstat_pit = np.mean(pp_vals <= obs_vals, axis=0)
            tstat_pit_dens, xmin, xmax = _fast_kde(tstat_pit)
            x_s = np.linspace(xmin, xmax, len(tstat_pit_dens))
            ax_i.plot(x_s, tstat_pit_dens, color=color)
            if reference is not None:
                if reference == "analytical":
                    n_obs = obs_vals.size
                    hdi = stats.beta(n_obs / 2, n_obs / 2).ppf((1 - hdi_prob) / 2)
                    hdi_odds = (hdi / (1 - hdi), (1 - hdi) / hdi)
                    ax_i.axhspan(*hdi_odds, **plot_ref_kwargs)
                elif reference == "samples":
                    dist = stats.uniform(0, 1)
                    x_ss, u_dens = sample_reference_distribution(dist, (tstat_pit_dens.size, n_ref))
                    ax_i.plot(x_ss, u_dens, linewidth=linewidth, **plot_ref_kwargs)
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
                ax_i.plot(0, 0, label=f"bpv={p_value:.2f}", alpha=0)
                ax_i.legend()

            if plot_mean:
                ax_i.plot(
                    obs_vals.mean(), 0, "o", color=color, markeredgecolor="k", markersize=markersize
                )

        if var_name != pp_var_name:
            xlabel = "{} / {}".format(var_name, pp_var_name)
        else:
            xlabel = var_name
        ax_i.set_title(make_label(xlabel, selection), fontsize=ax_labelsize)

    if backend_show(show):
        plt.show()

    return axes
