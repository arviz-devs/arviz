"""Bokeh Posterior predictive plot."""
import numpy as np
from bokeh.models.annotations import Legend

from ....stats.density_utils import get_bins, histogram, kde
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size, vectorized_to_hex

from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


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
    jitter,
    total_pp_samples,
    legend,  # pylint: disable=unused-argument
    labeller,
    group,  # pylint: disable=unused-argument
    animation_kwargs,  # pylint: disable=unused-argument
    num_pp_samples,
    backend_kwargs,
    show,
):
    """Bokeh ppc plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }

    colors = vectorized_to_hex(colors)

    (figsize, *_, linewidth, markersize) = _scale_fig_size(figsize, textsize, rows, cols)
    if ax is None:
        axes = create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
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
        raise ValueError("jitter must be >=0.")

    for i, ax_i in enumerate((item for item in axes.flatten() if item is not None)):
        var_name, sel, isel, obs_vals = obs_plotters[i]
        pp_var_name, _, _, pp_vals = pp_plotters[i]
        dtype = predictive_dataset[pp_var_name].dtype.kind
        legend_it = []

        # flatten non-specified dimensions
        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)
        pp_sampled_vals = pp_vals[pp_sample_ix]

        if kind == "kde":
            plot_kwargs = {
                "line_color": colors[0],
                "line_alpha": alpha,
                "line_width": 0.5 * linewidth,
            }

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

            if dtype == "f":
                multi_line = ax_i.multi_line(pp_xs, pp_densities, **plot_kwargs)
                legend_it.append((f"{group.capitalize()} predictive", [multi_line]))
            else:
                all_steps = []
                for x_s, y_s in zip(pp_xs, pp_densities):
                    step = ax_i.step(x_s, y_s, **plot_kwargs)
                    all_steps.append(step)
                legend_it.append((f"{group.capitalize()} predictive", all_steps))

            if observed:
                label = "Observed"
                if dtype == "f":
                    _, glyph = plot_kde(
                        obs_vals,
                        plot_kwargs={"line_color": colors[1], "line_width": linewidth},
                        fill_kwargs={"alpha": 0},
                        ax=ax_i,
                        backend="bokeh",
                        backend_kwargs={},
                        show=False,
                        return_glyph=True,
                    )
                    legend_it.append((label, glyph))
                else:
                    bins = get_bins(obs_vals)
                    _, hist, bin_edges = histogram(obs_vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    step = ax_i.step(
                        bin_edges,
                        hist,
                        line_color=colors[1],
                        line_width=linewidth,
                        mode="center",
                    )
                    legend_it.append((label, [step]))

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
                    line = ax_i.line(
                        new_x,
                        new_d.mean(0),
                        color=colors[2],
                        line_dash="dashed",
                        line_width=linewidth,
                    )
                    legend_it.append((label, [line]))
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    step = ax_i.step(
                        bin_edges,
                        hist,
                        line_color=colors[2],
                        line_width=linewidth,
                        line_dash="dashed",
                        mode="center",
                    )
                    legend_it.append((label, [step]))
            ax_i.yaxis.major_tick_line_color = None
            ax_i.yaxis.minor_tick_line_color = None
            ax_i.yaxis.major_label_text_font_size = "0pt"

        elif kind == "cumulative":
            if observed:
                label = "Observed"
                if dtype == "f":
                    glyph = ax_i.line(
                        *_empirical_cdf(obs_vals),
                        line_color=colors[1],
                        line_width=linewidth,
                    )
                    glyph.level = "overlay"
                    legend_it.append((label, [glyph]))

                else:
                    step = ax_i.step(
                        *_empirical_cdf(obs_vals),
                        line_color=colors[1],
                        line_width=linewidth,
                        mode="center",
                    )
                    legend_it.append((label, [step]))
            pp_densities = np.empty((2 * len(pp_sampled_vals), pp_sampled_vals[0].size))
            for idx, vals in enumerate(pp_sampled_vals):
                vals = np.array([vals]).flatten()
                pp_x, pp_density = _empirical_cdf(vals)
                pp_densities[2 * idx] = pp_x
                pp_densities[2 * idx + 1] = pp_density
            multi_line = ax_i.multi_line(
                list(pp_densities[::2]),
                list(pp_densities[1::2]),
                line_alpha=alpha,
                line_color=colors[0],
                line_width=linewidth,
            )
            legend_it.append((f"{group.capitalize()} predictive", [multi_line]))
            if mean:
                label = f"{group.capitalize()} predictive mean"
                line = ax_i.line(
                    *_empirical_cdf(pp_vals.flatten()),
                    color=colors[2],
                    line_dash="dashed",
                    line_width=linewidth,
                )
                legend_it.append((label, [line]))

        elif kind == "scatter":
            if mean:
                label = f"{group.capitalize()} predictive mean"
                if dtype == "f":
                    _, glyph = plot_kde(
                        pp_vals.flatten(),
                        plot_kwargs={
                            "line_color": colors[2],
                            "line_dash": "dashed",
                            "line_width": linewidth,
                        },
                        ax=ax_i,
                        backend="bokeh",
                        backend_kwargs={},
                        show=False,
                        return_glyph=True,
                    )
                    legend_it.append((label, glyph))
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    step = ax_i.step(
                        bin_edges,
                        hist,
                        color=colors[2],
                        line_width=linewidth,
                        line_dash="dashed",
                        mode="center",
                    )
                    legend_it.append((label, [step]))

            jitter_scale = 0.1
            y_rows = np.linspace(0, 0.1, num_pp_samples + 1)
            scale_low = 0
            scale_high = jitter_scale * jitter

            if observed:
                label = "Observed"
                obs_yvals = np.zeros_like(obs_vals, dtype=np.float64)
                if jitter:
                    obs_yvals += np.random.uniform(
                        low=scale_low, high=scale_high, size=len(obs_vals)
                    )
                glyph = ax_i.circle(
                    obs_vals,
                    obs_yvals,
                    line_color=colors[1],
                    fill_color=colors[1],
                    size=markersize,
                    line_alpha=alpha,
                )
                glyph.level = "overlay"
                legend_it.append((label, [glyph]))

            all_scatter = []
            for vals, y in zip(pp_sampled_vals, y_rows[1:]):
                vals = np.ravel(vals)
                yvals = np.full_like(vals, y, dtype=np.float64)
                if jitter:
                    yvals += np.random.uniform(low=scale_low, high=scale_high, size=len(vals))
                scatter = ax_i.scatter(
                    vals,
                    yvals,
                    line_color=colors[0],
                    fill_color=colors[0],
                    size=markersize,
                    fill_alpha=alpha,
                )
                all_scatter.append(scatter)

            legend_it.append((f"{group.capitalize()} predictive", all_scatter))
            ax_i.yaxis.major_tick_line_color = None
            ax_i.yaxis.minor_tick_line_color = None
            ax_i.yaxis.major_label_text_font_size = "0pt"

        if legend:
            legend = Legend(
                items=legend_it,
                location="top_left",
                orientation="vertical",
            )
            ax_i.add_layout(legend)
            if textsize is not None:
                ax_i.legend.label_text_font_size = f"{textsize}pt"
            ax_i.legend.click_policy = "hide"
        ax_i.xaxis.axis_label = labeller.make_pp_label(var_name, pp_var_name, sel, isel)

    show_layout(axes, show)

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
