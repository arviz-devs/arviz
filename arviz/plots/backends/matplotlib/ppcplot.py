"""Matplotib Posterior predictive plot."""
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ...kdeplot import plot_kde, _fast_kde
from ...plot_utils import (
    make_label,
    _create_axes_grid,
    get_bins,
)
from ....stats.stats_utils import histogram


def plot_ppc(
    ax,
    length_plotters,
    rows,
    cols,
    figsize,
    animated,
    obs_plotters,
    pp_plotters,
    posterior_predictive,
    pp_sample_ix,
    kind,
    alpha,
    linewidth,
    mean,
    xt_labelsize,
    ax_labelsize,
    jitter,
    total_pp_samples,
    legend,
    markersize,
    animation_kwargs,
    num_pp_samples,
    backend_kwargs,
    show,
):
    """Matplotlib ppc plot."""
    if ax is None:
        fig, axes = _create_axes_grid(
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
        if animated:
            fig = axes[0].get_figure()
            if not all([ax.get_figure() is fig for ax in axes]):
                raise ValueError("All axes must be on the same figure for animation to work")

    for i, ax_i in enumerate(axes):
        var_name, selection, obs_vals = obs_plotters[i]
        pp_var_name, _, pp_vals = pp_plotters[i]
        dtype = posterior_predictive[pp_var_name].dtype.kind

        # flatten non-specified dimensions
        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)
        pp_sampled_vals = pp_vals[pp_sample_ix]

        if kind == "kde":
            plot_kwargs = {"color": "C5", "alpha": alpha, "linewidth": 0.5 * linewidth}
            if dtype == "i":
                plot_kwargs["drawstyle"] = "steps-pre"
            ax_i.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))

            if dtype == "f":
                plot_kde(
                    obs_vals,
                    label="Observed {}".format(var_name),
                    plot_kwargs={"color": "k", "linewidth": linewidth, "zorder": 3},
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
                    label="Observed {}".format(var_name),
                    color="k",
                    linewidth=linewidth,
                    zorder=3,
                    drawstyle=plot_kwargs["drawstyle"],
                )

            pp_densities = []
            pp_xs = []
            for vals in pp_sampled_vals:
                vals = np.array([vals]).flatten()
                if dtype == "f":
                    pp_density, lower, upper = _fast_kde(vals)
                    pp_x = np.linspace(lower, upper, len(pp_density))
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
                        color="C0",
                        linestyle="--",
                        linewidth=linewidth,
                        zorder=2,
                        label="Posterior predictive mean {}".format(pp_var_name),
                    )
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    ax_i.plot(
                        bin_edges,
                        hist,
                        color="C0",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name),
                        zorder=2,
                        linestyle="--",
                        drawstyle=plot_kwargs["drawstyle"],
                    )
            ax_i.tick_params(labelsize=xt_labelsize)
            ax_i.set_yticks([])

        elif kind == "cumulative":
            drawstyle = "default" if dtype == "f" else "steps-pre"
            ax_i.plot(
                *_empirical_cdf(obs_vals),
                color="k",
                linewidth=linewidth,
                label="Observed {}".format(var_name),
                drawstyle=drawstyle,
                zorder=3
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
                    *pp_densities, alpha=alpha, color="C5", drawstyle=drawstyle, linewidth=linewidth
                )
            ax_i.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))
            if mean:
                ax_i.plot(
                    *_empirical_cdf(pp_vals.flatten()),
                    color="C0",
                    linestyle="--",
                    linewidth=linewidth,
                    drawstyle=drawstyle,
                    label="Posterior predictive mean {}".format(pp_var_name)
                )
            ax_i.set_yticks([0, 0.5, 1])

        elif kind == "scatter":
            if mean:
                if dtype == "f":
                    plot_kde(
                        pp_vals.flatten(),
                        plot_kwargs={
                            "color": "C0",
                            "linestyle": "--",
                            "linewidth": linewidth,
                            "zorder": 3,
                        },
                        label="Posterior predictive mean {}".format(pp_var_name),
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
                        color="C0",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name),
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

            obs_yvals = np.zeros_like(obs_vals, dtype=np.float64)
            if jitter:
                obs_yvals += np.random.uniform(low=scale_low, high=scale_high, size=len(obs_vals))
            ax_i.plot(
                obs_vals,
                obs_yvals,
                "o",
                color="C0",
                markersize=markersize,
                alpha=alpha,
                label="Observed {}".format(var_name),
                zorder=4,
            )

            if animated:
                animate, init = _set_animation(
                    pp_sampled_vals,
                    ax_i,
                    kind=kind,
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
                        vals, yvals, "o", zorder=2, color="C5", markersize=markersize, alpha=alpha
                    )

            ax_i.plot([], "C5o", label="Posterior predictive {}".format(pp_var_name))

            ax_i.set_yticks([])

        if var_name != pp_var_name:
            xlabel = "{} / {}".format(var_name, pp_var_name)
        else:
            xlabel = var_name
        ax_i.set_xlabel(make_label(xlabel, selection), fontsize=ax_labelsize)

        if legend:
            if i == 0:
                ax_i.legend(fontsize=xt_labelsize * 0.75)
            else:
                ax_i.legend([])

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
    drawstyle=None,
    linewidth=None,
    height=None,
    markersize=None,
    plot_kwargs=None,
):
    if kind == "kde":
        length = len(pp_sampled_vals)
        if dtype == "f":
            y_vals, lower, upper = _fast_kde(pp_sampled_vals[0])
            x_vals = np.linspace(lower, upper, len(y_vals))

            max_max = max([max(_fast_kde(pp_sampled_vals[i])[0]) for i in range(length)])

            ax.set_ylim(0, max_max)

            (line,) = ax.plot(x_vals, y_vals, **plot_kwargs)

            def animate(i):
                y_vals, lower, upper = _fast_kde(pp_sampled_vals[i])
                x_vals = np.linspace(lower, upper, len(y_vals))
                line.set_data(x_vals, y_vals)
                return line

        else:
            vals = pp_sampled_vals[0]
            _, y_vals, x_vals = histogram(vals, bins="auto")
            (line,) = ax.plot(x_vals[:-1], y_vals, **plot_kwargs)

            max_max = max(
                [max(histogram(pp_sampled_vals[i], bins="auto")[1]) for i in range(length)]
            )

            ax.set_ylim(0, max_max)

            def animate(i):
                _, y_vals, x_vals = histogram(pp_sampled_vals[i], bins="auto")
                line.set_data(x_vals[:-1], y_vals)
                return (line,)

    elif kind == "cumulative":
        x_vals, y_vals = _empirical_cdf(pp_sampled_vals[0])
        (line,) = ax.plot(
            x_vals, y_vals, alpha=alpha, color="C5", drawstyle=drawstyle, linewidth=linewidth
        )

        def animate(i):
            x_vals, y_vals = _empirical_cdf(pp_sampled_vals[i])
            line.set_data(x_vals, y_vals)
            return line

    elif kind == "scatter":
        x_vals = pp_sampled_vals[0]
        y_vals = np.full_like(x_vals, height, dtype=np.float64)
        (line,) = ax.plot(
            x_vals, y_vals, "o", zorder=2, color="C5", markersize=markersize, alpha=alpha
        )

        def animate(i):
            line.set_xdata(np.ravel(pp_sampled_vals[i]))
            return line

    def init():
        if kind != "scatter":
            line.set_data([], [])
        else:
            line.set_xdata([])
        return line

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
