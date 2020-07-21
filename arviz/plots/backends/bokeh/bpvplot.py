"""Bokeh Bayesian p-value Posterior predictive plot."""
import numpy as np
from bokeh.models import BoxAnnotation
from bokeh.models.annotations import Title
from matplotlib.colors import to_hex
from scipy import stats

from . import backend_kwarg_defaults
from .. import show_layout
from ...kdeplot import plot_kde
from ...plot_utils import (
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
    """Bokeh bpv plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi"),),
        **backend_kwargs,
    }

    if plot_ref_kwargs is None:
        plot_ref_kwargs = {}
    if kind == "p_value" and reference == "analytical":
        plot_ref_kwargs.setdefault("color", "black")
        plot_ref_kwargs.setdefault("line_dash", "dashed")
    else:
        plot_ref_kwargs.setdefault("alpha", 0.1)
        plot_ref_kwargs.setdefault("color", to_hex(color))

    (figsize, ax_labelsize, _, _, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    if ax is None:
        _, axes = _create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            backend="bokeh",
            backend_kwargs=backend_kwargs,
        )
    else:
        axes = np.atleast_2d(ax)

        if len([item for item in axes.ravel() if not None]) != length_plotters:
            raise ValueError(
                "Found {} variables to plot but {} axes instances. They must be equal.".format(
                    length_plotters, len(axes)
                )
            )

    for i, ax_i in enumerate((item for item in axes.flatten() if item is not None)):
        var_name, _, obs_vals = obs_plotters[i]
        pp_var_name, _, pp_vals = pp_plotters[i]

        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)

        if kind == "p_value":
            tstat_pit = np.mean(pp_vals <= obs_vals, axis=-1)
            tstat_pit_dens, xmin, xmax = _fast_kde(tstat_pit)
            x_s = np.linspace(xmin, xmax, len(tstat_pit_dens))
            ax_i.line(x_s, tstat_pit_dens, line_width=linewidth, color=color)
            # ax_i.set_yticks([])
            if reference is not None:
                dist = stats.beta(obs_vals.size / 2, obs_vals.size / 2)
                if reference == "analytical":
                    lwb = dist.ppf((1 - 0.9999) / 2)
                    upb = 1 - lwb
                    x = np.linspace(lwb, upb, 500)
                    dens_ref = dist.pdf(x)
                    ax_i.line(x, dens_ref, **plot_ref_kwargs)
                elif reference == "samples":
                    x_ss, u_dens = sample_reference_distribution(
                        dist, (n_ref, tstat_pit_dens.size,)
                    )
                    for x_ss_i, u_dens_i in zip(x_ss.T, u_dens.T):
                        ax_i.line(x_ss_i, u_dens_i, line_width=linewidth, **plot_ref_kwargs)

        elif kind == "u_value":
            tstat_pit = np.mean(pp_vals <= obs_vals, axis=0)
            tstat_pit_dens, xmin, xmax = _fast_kde(tstat_pit)
            x_s = np.linspace(xmin, xmax, len(tstat_pit_dens))
            ax_i.line(x_s, tstat_pit_dens, color=color)
            if reference is not None:
                if reference == "analytical":
                    n_obs = obs_vals.size
                    hdi = stats.beta(n_obs / 2, n_obs / 2).ppf((1 - hdi_prob) / 2)
                    hdi_odds = (hdi / (1 - hdi), (1 - hdi) / hdi)
                    ax_i.add_layout(
                        BoxAnnotation(
                            bottom=hdi_odds[1],
                            top=hdi_odds[0],
                            fill_alpha=plot_ref_kwargs.pop("alpha"),
                            fill_color=plot_ref_kwargs.pop("color"),
                            **plot_ref_kwargs,
                        )
                    )

                elif reference == "samples":
                    dist = stats.uniform(0, 1)
                    x_ss, u_dens = sample_reference_distribution(dist, (tstat_pit_dens.size, n_ref))
                    for x_ss_i, u_dens_i in zip(x_ss.T, u_dens.T):
                        ax_i.line(x_ss_i, u_dens_i, line_width=linewidth, **plot_ref_kwargs)
            # ax_i.set_ylim(0, None)
            # ax_i.set_xlim(0, 1)
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

            plot_kde(pp_vals, ax=ax_i, plot_kwargs={"color": color}, backend="bokeh", show=False)
            # ax_i.set_yticks([])
            if bpv:
                p_value = np.mean(pp_vals <= obs_vals)
                ax_i.line(0, 0, legend_label=f"bpv={p_value:.2f}", alpha=0)

            if plot_mean:
                ax_i.circle(
                    obs_vals.mean(), 0, fill_color=color, line_color="black", size=markersize
                )

        if var_name != pp_var_name:
            xlabel = "{} / {}".format(var_name, pp_var_name)
        else:
            xlabel = var_name
        _title = Title()
        _title.text = xlabel
        ax_i.title = _title
        size = str(int(ax_labelsize))
        ax_i.title.text_font_size = f"{size}pt"

    show_layout(axes, show)

    return axes
