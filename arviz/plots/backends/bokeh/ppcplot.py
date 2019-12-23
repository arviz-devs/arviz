"""Bokeh Posterior predictive plot."""
import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot

from . import backend_kwarg_defaults, backend_show
from ...kdeplot import plot_kde, _fast_kde
from ...plot_utils import (
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
    obs_plotters,
    pp_plotters,
    posterior_predictive,
    pp_sample_ix,
    kind,
    alpha,
    linewidth,
    mean,
    jitter,
    total_pp_samples,
    markersize,
    backend_kwargs,
    num_pp_samples,
    show,
):
    """Bokeh ppc plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
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
        dtype = posterior_predictive[pp_var_name].dtype.kind

        # flatten non-specified dimensions
        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)
        pp_sampled_vals = pp_vals[pp_sample_ix]

        if kind == "kde":
            plot_kwargs = {"line_color": "red", "line_alpha": alpha, "line_width": 0.5 * linewidth}

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

            if dtype == "f":
                ax_i.multi_line(pp_xs, pp_densities, **plot_kwargs)
            else:
                for x_s, y_s in zip(pp_xs, pp_densities):
                    ax_i.step(x_s, y_s, **plot_kwargs)

            if dtype == "f":
                plot_kde(
                    obs_vals,
                    plot_kwargs={"line_color": "black", "line_width": linewidth},
                    fill_kwargs={"alpha": 0},
                    ax=ax_i,
                    backend="bokeh",
                    backend_kwargs={},
                    show=False,
                )
            else:
                bins = get_bins(obs_vals)
                _, hist, bin_edges = histogram(obs_vals, bins=bins)
                hist = np.concatenate((hist[:1], hist))
                ax_i.step(
                    bin_edges, hist, line_color="black", line_width=linewidth, mode="center",
                )

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
                    ax_i.line(
                        new_x,
                        new_d.mean(0),
                        color="blue",
                        line_dash="dashed",
                        line_width=linewidth,
                    )
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    ax_i.step(
                        bin_edges,
                        hist,
                        line_color="blue",
                        line_width=linewidth,
                        line_dash="dashed",
                        mode="center",
                    )
            ax_i.yaxis.major_tick_line_color = None
            ax_i.yaxis.minor_tick_line_color = None
            ax_i.yaxis.major_label_text_font_size = "0pt"

        elif kind == "cumulative":
            if dtype == "f":
                ax_i.line(
                    *_empirical_cdf(obs_vals), line_color="black", line_width=linewidth,
                )
            else:
                ax_i.step(
                    *_empirical_cdf(obs_vals),
                    line_color="black",
                    line_width=linewidth,
                    mode="center",
                )
            pp_densities = np.empty((2 * len(pp_sampled_vals), pp_sampled_vals[0].size))
            for idx, vals in enumerate(pp_sampled_vals):
                vals = np.array([vals]).flatten()
                pp_x, pp_density = _empirical_cdf(vals)
                pp_densities[2 * idx] = pp_x
                pp_densities[2 * idx + 1] = pp_density
            ax_i.multi_line(
                list(pp_densities[::2]),
                list(pp_densities[1::2]),
                line_alpha=alpha,
                line_color="pink",
                line_width=linewidth,
            )
            if mean:
                ax_i.line(
                    *_empirical_cdf(pp_vals.flatten()),
                    color="blue",
                    line_dash="dashed",
                    line_width=linewidth,
                )

        elif kind == "scatter":
            if mean:
                if dtype == "f":
                    plot_kde(
                        pp_vals.flatten(),
                        plot_kwargs={
                            "line_color": "blue",
                            "line_dash": "dashed",
                            "line_width": linewidth,
                        },
                        ax=ax_i,
                        backend="bokeh",
                        backend_kwargs={},
                        show=False,
                    )
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    ax_i.step(
                        bin_edges,
                        hist,
                        color="blue",
                        line_width=linewidth,
                        line_dash="dashed",
                        mode="center",
                    )

            jitter_scale = 0.1
            y_rows = np.linspace(0, 0.1, num_pp_samples + 1)
            scale_low = 0
            scale_high = jitter_scale * jitter

            obs_yvals = np.zeros_like(obs_vals, dtype=np.float64)
            if jitter:
                obs_yvals += np.random.uniform(low=scale_low, high=scale_high, size=len(obs_vals))
            ax_i.circle(
                obs_vals, obs_yvals, fill_color="black", size=markersize, line_alpha=alpha,
            )

            for vals, y in zip(pp_sampled_vals, y_rows[1:]):
                vals = np.ravel(vals)
                yvals = np.full_like(vals, y, dtype=np.float64)
                if jitter:
                    yvals += np.random.uniform(low=scale_low, high=scale_high, size=len(vals))
                ax_i.circle(vals, yvals, fill_color="red", size=markersize, fill_alpha=alpha)

            ax_i.yaxis.major_tick_line_color = None
            ax_i.yaxis.minor_tick_line_color = None
            ax_i.yaxis.major_label_text_font_size = "0pt"

        if var_name != pp_var_name:
            xlabel = "{} / {}".format(var_name, pp_var_name)
        else:
            xlabel = var_name
        ax_i.xaxis.axis_label = xlabel

    if backend_show(show):
        grid = gridplot(axes.tolist(), toolbar_location="above")
        bkp.show(grid)

    return axes


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
