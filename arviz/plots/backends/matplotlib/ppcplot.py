"""Matplotlib Posterior predictive plot."""

import logging
import platform

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, get_backend

from ....stats.density_utils import get_bins, histogram, kde
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid

_log = logging.getLogger(__name__)


def plot_ppc(
    ax,
    length_plotters,
    rows,
    cols,
    figsize,
    animated,
    obs_plotters,
    pp_plotters,
    predictive_dataset,
    pp_sample_ix,
    kind,
    alpha,
    colors,
    textsize,
    mean,
    observed,
    observed_rug,
    jitter,
    total_pp_samples,
    legend,
    labeller,
    group,
    animation_kwargs,
    num_pp_samples,
    backend_kwargs,
    show,
):
    """Matplotlib ppc plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if animation_kwargs is None:
        animation_kwargs = {}
    if platform.system() == "Linux":
        animation_kwargs.setdefault("blit", True)
    else:
        animation_kwargs.setdefault("blit", False)

    if alpha is None:
        if animated:
            alpha = 1
        else:
            if kind.lower() == "scatter":
                alpha = 0.7
            else:
                alpha = 0.2

    if jitter is None:
        jitter = 0.0
    if jitter < 0.0:
        raise ValueError("jitter must be >=0")

    if animated:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell" and get_backend() != "nbAgg":
                raise Warning(
                    "To run animations inside a notebook you have to use the nbAgg backend. "
                    "Try with `%matplotlib notebook` or  `%matplotlib  nbAgg`. You can switch "
                    "back to the default backend with `%matplotlib  inline` or "
                    "`%matplotlib  auto`."
                )
        except NameError:
            pass

        if animation_kwargs["blit"] and platform.system() != "Linux":
            _log.warning(
                "If you experience problems rendering the animation try setting "
                "`animation_kwargs({'blit':False}) or changing the plotting backend "
                "(e.g. to TkAgg)"
            )

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("squeeze", True)
    if ax is None:
        fig, axes = create_axes_grid(length_plotters, rows, cols, backend_kwargs=backend_kwargs)
    else:
        axes = np.ravel(ax)
        if len(axes) != length_plotters:
            raise ValueError(
                "Found {} variables to plot but {} axes instances. They must be equal.".format(
                    length_plotters, len(axes)
                )
            )
        if animated:
            fig = axes[0].get_figure()
            if not all((ax.get_figure() is fig for ax in axes)):
                raise ValueError("All axes must be on the same figure for animation to work")

    for i, ax_i in enumerate(np.ravel(axes)[:length_plotters]):
        var_name, selection, isel, obs_vals = obs_plotters[i]
        pp_var_name, _, _, pp_vals = pp_plotters[i]
        dtype = predictive_dataset[pp_var_name].dtype.kind

        if dtype not in ["i", "f"]:
            raise ValueError(
                f"The data type of the predictive data must be one of 'i' or 'f', but is '{dtype}'"
            )

        # flatten non-specified dimensions
        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)
        pp_sampled_vals = pp_vals[pp_sample_ix]

        if kind == "kde":
            plot_kwargs = {"color": colors[0], "alpha": alpha, "linewidth": 0.5 * linewidth}
            if dtype == "i":
                plot_kwargs["drawstyle"] = "steps-pre"
            ax_i.plot([], color=colors[0], label=f"{group.capitalize()} predictive")
            if observed:
                if dtype == "f":
                    plot_kde(
                        obs_vals,
                        rug=observed_rug,
                        label="Observed",
                        plot_kwargs={"color": colors[1], "linewidth": linewidth, "zorder": 3},
                        fill_kwargs={"alpha": 0},
                        ax=ax_i,
                        legend=legend,
                    )
                else:
                    bins = get_bins(obs_vals)
                    _, hist, bin_edges = histogram(obs_vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    ax_i.plot(
                        bin_edges,
                        hist,
                        label="Observed",
                        color=colors[1],
                        linewidth=linewidth,
                        zorder=3,
                        drawstyle=plot_kwargs["drawstyle"],
                    )

            pp_densities = []
            pp_xs = []
            for vals in pp_sampled_vals:
                vals = np.array([vals]).flatten()
                if dtype == "f":
                    pp_x, pp_density = kde(vals)
                    pp_densities.append(pp_density)
                    pp_xs.append(pp_x)
                else:
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    pp_densities.append(hist)
                    pp_xs.append(bin_edges)

            if animated:
                animate, init = _set_animation(
                    pp_sampled_vals, ax_i, dtype=dtype, kind=kind, plot_kwargs=plot_kwargs
                )

            else:
                if dtype == "f":
                    ax_i.plot(np.transpose(pp_xs), np.transpose(pp_densities), **plot_kwargs)
                else:
                    for x_s, y_s in zip(pp_xs, pp_densities):
                        ax_i.plot(x_s, y_s, **plot_kwargs)

            if mean:
                label = f"{group.capitalize()} predictive mean"
                if dtype == "f":
                    rep = len(pp_densities)
                    len_density = len(pp_densities[0])

                    new_x = np.linspace(np.min(pp_xs), np.max(pp_xs), len_density)
                    new_d = np.zeros((rep, len_density))
                    bins = np.digitize(pp_xs, new_x, right=True)
                    new_x -= (new_x[1] - new_x[0]) / 2
                    for irep in range(rep):
                        new_d[irep][bins[irep]] = pp_densities[irep]
                    ax_i.plot(
                        new_x,
                        new_d.mean(0),
                        color=colors[2],
                        linestyle="--",
                        linewidth=linewidth * 1.5,
                        zorder=2,
                        label=label,
                    )
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    ax_i.plot(
                        bin_edges,
                        hist,
                        color=colors[2],
                        linewidth=linewidth * 1.5,
                        label=label,
                        zorder=2,
                        linestyle="--",
                        drawstyle=plot_kwargs["drawstyle"],
                    )
            ax_i.tick_params(labelsize=xt_labelsize)
            ax_i.set_yticks([])

        elif kind == "cumulative":
            drawstyle = "default" if dtype == "f" else "steps-pre"
            if observed:
                ax_i.plot(
                    *_empirical_cdf(obs_vals),
                    color=colors[1],
                    linewidth=linewidth,
                    label="Observed",
                    drawstyle=drawstyle,
                    zorder=3,
                )
                if observed_rug:
                    ax_i.plot(
                        obs_vals,
                        np.zeros_like(obs_vals) - 0.1,
                        ls="",
                        marker="|",
                        color=colors[1],
                    )
            if animated:
                animate, init = _set_animation(
                    pp_sampled_vals,
                    ax_i,
                    kind=kind,
                    alpha=alpha,
                    drawstyle=drawstyle,
                    linewidth=linewidth,
                )

            else:
                pp_densities = np.empty((2 * len(pp_sampled_vals), pp_sampled_vals[0].size))
                for idx, vals in enumerate(pp_sampled_vals):
                    vals = np.array([vals]).flatten()
                    pp_x, pp_density = _empirical_cdf(vals)
                    pp_densities[2 * idx] = pp_x
                    pp_densities[2 * idx + 1] = pp_density

                ax_i.plot(
                    *pp_densities,
                    alpha=alpha,
                    color=colors[0],
                    drawstyle=drawstyle,
                    linewidth=linewidth,
                )
            ax_i.plot([], color=colors[0], label=f"{group.capitalize()} predictive")
            if mean:
                ax_i.plot(
                    *_empirical_cdf(pp_vals.flatten()),
                    color=colors[2],
                    linestyle="--",
                    linewidth=linewidth * 1.5,
                    drawstyle=drawstyle,
                    label=f"{group.capitalize()} predictive mean",
                )
            ax_i.set_yticks([0, 0.5, 1])

        elif kind == "scatter":
            if mean:
                if dtype == "f":
                    plot_kde(
                        pp_vals.flatten(),
                        plot_kwargs={
                            "color": colors[2],
                            "linestyle": "--",
                            "linewidth": linewidth * 1.5,
                            "zorder": 3,
                        },
                        label=f"{group.capitalize()} predictive mean",
                        ax=ax_i,
                        legend=legend,
                    )
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    ax_i.plot(
                        bin_edges,
                        hist,
                        color=colors[2],
                        linewidth=linewidth * 1.5,
                        label=f"{group.capitalize()} predictive mean",
                        zorder=3,
                        linestyle="--",
                        drawstyle="steps-pre",
                    )

            _, limit = ax_i.get_ylim()
            limit *= 1.05
            y_rows = np.linspace(0, limit, num_pp_samples + 1)
            jitter_scale = y_rows[1] - y_rows[0]
            scale_low = 0
            scale_high = jitter_scale * jitter

            if observed:
                obs_yvals = np.zeros_like(obs_vals, dtype=np.float64)
                if jitter:
                    obs_yvals += np.random.uniform(
                        low=scale_low, high=scale_high, size=len(obs_vals)
                    )
                ax_i.plot(
                    obs_vals,
                    obs_yvals,
                    "o",
                    color=colors[1],
                    markersize=markersize,
                    alpha=alpha,
                    label="Observed",
                    zorder=4,
                )

            if animated:
                animate, init = _set_animation(
                    pp_sampled_vals,
                    ax_i,
                    kind=kind,
                    color=colors[0],
                    height=y_rows.mean() * 0.5,
                    markersize=markersize,
                )

            else:
                for vals, y in zip(pp_sampled_vals, y_rows[1:]):
                    vals = np.ravel(vals)
                    yvals = np.full_like(vals, y, dtype=np.float64)
                    if jitter:
                        yvals += np.random.uniform(low=scale_low, high=scale_high, size=len(vals))
                    ax_i.plot(
                        vals,
                        yvals,
                        "o",
                        zorder=2,
                        color=colors[0],
                        markersize=markersize,
                        alpha=alpha,
                    )

            ax_i.plot([], color=colors[0], marker="o", label=f"{group.capitalize()} predictive")

            ax_i.set_yticks([])

        ax_i.set_xlabel(
            labeller.make_pp_label(var_name, pp_var_name, selection, isel), fontsize=ax_labelsize
        )

        if legend:
            if i == 0:
                ax_i.legend(fontsize=xt_labelsize * 0.75)

    if backend_show(show):
        plt.show()

    if animated:
        ani = animation.FuncAnimation(
            fig, animate, np.arange(0, num_pp_samples), init_func=init, **animation_kwargs
        )
        return axes, ani
    else:
        return axes


def _set_animation(
    pp_sampled_vals,
    ax,
    dtype=None,
    kind="density",
    alpha=None,
    color=None,
    drawstyle=None,
    linewidth=None,
    height=None,
    markersize=None,
    plot_kwargs=None,
):
    if kind == "kde":
        length = len(pp_sampled_vals)
        if dtype == "f":
            x_vals, y_vals = kde(pp_sampled_vals[0])
            max_max = max(max(kde(pp_sampled_vals[i])[1]) for i in range(length))
            ax.set_ylim(0, max_max)
            (line,) = ax.plot(x_vals, y_vals, **plot_kwargs)

            def animate(i):
                x_vals, y_vals = kde(pp_sampled_vals[i])
                line.set_data(x_vals, y_vals)
                return (line,)

        else:
            vals = pp_sampled_vals[0]
            bins = get_bins(vals)
            _, y_vals, x_vals = histogram(vals, bins=bins)
            (line,) = ax.plot(x_vals[:-1], y_vals, **plot_kwargs)

            max_max = max(
                max(histogram(pp_sampled_vals[i], bins=get_bins(pp_sampled_vals[i]))[1])
                for i in range(length)
            )

            ax.set_ylim(0, max_max)

            def animate(i):
                pp_vals = pp_sampled_vals[i]
                _, y_vals, x_vals = histogram(pp_vals, bins=get_bins(pp_vals))
                line.set_data(x_vals[:-1], y_vals)
                return (line,)

    elif kind == "cumulative":
        x_vals, y_vals = _empirical_cdf(pp_sampled_vals[0])
        (line,) = ax.plot(
            x_vals, y_vals, alpha=alpha, color=color, drawstyle=drawstyle, linewidth=linewidth
        )

        def animate(i):
            x_vals, y_vals = _empirical_cdf(pp_sampled_vals[i])
            line.set_data(x_vals, y_vals)
            return (line,)

    elif kind == "scatter":
        x_vals = pp_sampled_vals[0]
        y_vals = np.full_like(x_vals, height, dtype=np.float64)
        (line,) = ax.plot(
            x_vals, y_vals, "o", zorder=2, color=color, markersize=markersize, alpha=alpha
        )

        def animate(i):
            line.set_xdata(np.ravel(pp_sampled_vals[i]))
            return (line,)

    def init():
        if kind != "scatter":
            line.set_data([], [])
        else:
            line.set_xdata([])
        return (line,)

    return animate, init


def _empirical_cdf(data):
    """Compute empirical cdf of a numpy array.

    Parameters
    ----------
    data : np.array
        1d array

    Returns
    -------
    np.array, np.array
        x and y coordinates for the empirical cdf of the data
    """
    return np.sort(data), np.linspace(0, 1, len(data))
