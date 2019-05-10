"""Posterior predictive plot."""
from numbers import Integral
import platform
import logging
import numpy as np
from matplotlib import animation
from .kdeplot import plot_kde, _fast_kde
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    default_grid,
    make_label,
    _create_axes_grid,
)
from ..utils import _var_names

_log = logging.getLogger(__name__)


def plot_ppc(
    data,
    kind="density",
    alpha=None,
    mean=True,
    figsize=None,
    textsize=None,
    data_pairs=None,
    var_names=None,
    coords=None,
    flatten=None,
    flatten_pp=None,
    num_pp_samples=None,
    random_seed=None,
    jitter=None,
    animated=False,
    animation_kwargs=None,
    legend=True,
):
    """
    Plot for posterior predictive checks.

    Parameters
    ----------
    data : az.InferenceData object
        InferenceData object containing the observed and posterior
        predictive data.
    kind : str
        Type of plot to display (density, cumulative, or scatter). Defaults to density.
    alpha : float
        Opacity of posterior predictive density curves. Defaults to 0.2 for kind = density
        and cumulative, for scatter defaults to 0.7
    mean : bool
        Whether or not to plot the mean posterior predictive distribution. Defaults to True
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.
    data_pairs : dict
        Dictionary containing relations between observed data and posterior predictive data.
        Dictionary structure:
        Key = data var_name
        Value = posterior predictive var_name
        For example, `data_pairs = {'y' : 'y_hat'}`
        If None, it will assume that the observed data and the posterior
        predictive data have the same variable name.
    var_names : list
        List of variables to be plotted. Defaults to all observed variables in the
        model if None.
    coords : dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. Defaults to including all coordinates for all
        dimensions if None.
    flatten : list
        List of dimensions to flatten in observed_data. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
    flatten_pp : list
        List of dimensions to flatten in posterior_predictive. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
        Dimensions should match flatten excluding dimensions for data_pairs parameters.
        If flatten is defined and flatten_pp is None, then `flatten_pp=flatten`.
    num_pp_samples : int
        The number of posterior predictive samples to plot. For `kind` = 'scatter' and
        `animation = False` if defaults to a maximum of 5 samples and will set jitter to 0.7
        unless defined otherwise. Otherwise it defaults to all provided samples.
    random_seed : int
        Random number generator seed passed to numpy.random.seed to allow
        reproducibility of the plot. By default, no seed will be provided
        and the plot will change each call if a random sample is specified
        by `num_pp_samples`.
    jitter : float
        If kind is "scatter", jitter will add random uniform noise to the height
        of the ppc samples and observed data. By default 0.
    animated : bool
        Create an animation of one posterior predictive sample per frame. Defaults to False.
    animation_kwargs : dict
        Keywords passed to `animation.FuncAnimation`.
    legend : bool
        Add legend to figure. By default True.

    Returns
    -------
    axes : matplotlib axes

    Examples
    --------
    Plot the observed data KDE overlaid on posterior predictive KDEs.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('radon')
        >>> az.plot_ppc(data)

    Plot the overlay with empirical CDFs.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, kind='cumulative')

    Use the coords and flatten parameters to plot selected variable dimensions
    across multiple plots.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, coords={'observed_county': ['ANOKA', 'BELTRAMI']}, flatten=[])

    Plot the overlay using a stacked scatter plot that is particularly useful
    when the sample sizes are small.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, kind='scatter', flatten=[],
        >>>             coords={'observed_county': ['AITKIN', 'BELTRAMI']})

    Plot random posterior predictive sub-samples.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, num_pp_samples=30, random_seed=7)
    """
    for group in ("posterior_predictive", "observed_data"):
        if not hasattr(data, group):
            raise TypeError(
                '`data` argument must have the group "{group}" for ppcplot'.format(group=group)
            )

    if kind.lower() not in ("density", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `density`, `cumulative`, or `scatter`")

    if data_pairs is None:
        data_pairs = {}

    if animation_kwargs is None:
        animation_kwargs = {}
    if platform.system() == "Linux":
        animation_kwargs.setdefault("blit", True)
    else:
        animation_kwargs.setdefault("blit", False)

    if animated and animation_kwargs["blit"] and platform.system() != "Linux":
        _log.warning(
            "If you experience problems rendering the animation try setting"
            "`animation_kwargs({'blit':False}) or changing the plotting backend (e.g. to TkAgg)"
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
    assert jitter >= 0.0

    observed = data.observed_data
    posterior_predictive = data.posterior_predictive

    if var_names is None:
        var_names = observed.data_vars
    var_names = _var_names(var_names, observed)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]

    if flatten_pp is None and flatten is None:
        flatten_pp = list(posterior_predictive.dims.keys())
    elif flatten_pp is None:
        flatten_pp = flatten
    if flatten is None:
        flatten = list(observed.dims.keys())

    if coords is None:
        coords = {}

    if random_seed is not None:
        np.random.seed(random_seed)

    total_pp_samples = posterior_predictive.sizes["chain"] * posterior_predictive.sizes["draw"]
    if num_pp_samples is None:
        if kind == "scatter" and not animated:
            num_pp_samples = min(5, total_pp_samples)
        else:
            num_pp_samples = total_pp_samples

    if (
        not isinstance(num_pp_samples, Integral)
        or num_pp_samples < 1
        or num_pp_samples > total_pp_samples
    ):
        raise TypeError(
            "`num_pp_samples` must be an integer between 1 and "
            + "{limit}.".format(limit=total_pp_samples)
        )

    pp_sample_ix = np.random.choice(total_pp_samples, size=num_pp_samples, replace=False)

    for key in coords.keys():
        coords[key] = np.where(np.in1d(observed[key], coords[key]))[0]

    obs_plotters = list(
        xarray_var_iter(
            observed.isel(coords), skip_dims=set(flatten), var_names=var_names, combined=True
        )
    )
    pp_plotters = list(
        xarray_var_iter(
            posterior_predictive.isel(coords),
            var_names=pp_var_names,
            skip_dims=set(flatten_pp),
            combined=True,
        )
    )
    length_plotters = len(obs_plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    fig, axes = _create_axes_grid(length_plotters, rows, cols, figsize=figsize)

    for i, ax in enumerate(axes):
        var_name, selection, obs_vals = obs_plotters[i]
        pp_var_name, _, pp_vals = pp_plotters[i]
        dtype = posterior_predictive[pp_var_name].dtype.kind

        # flatten non-specified dimensions
        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)
        pp_sampled_vals = pp_vals[pp_sample_ix]

        if kind == "density":
            plot_kwargs = {"color": "C5", "alpha": alpha, "linewidth": 0.5 * linewidth}
            if dtype == "i":
                plot_kwargs["drawstyle"] = "steps-pre"
            ax.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))

            if dtype == "f":
                plot_kde(
                    obs_vals,
                    label="Observed {}".format(var_name),
                    plot_kwargs={"color": "k", "linewidth": linewidth, "zorder": 3},
                    fill_kwargs={"alpha": 0},
                    ax=ax,
                    legend=legend,
                )
            else:
                nbins = round(len(obs_vals) ** 0.5)
                hist, bin_edges = np.histogram(obs_vals, bins=nbins, density=True)
                hist = np.concatenate((hist[:1], hist))
                ax.plot(
                    bin_edges,
                    hist,
                    label="Observed {}".format(var_name),
                    color="k",
                    linewidth=linewidth,
                    zorder=3,
                    drawstyle=plot_kwargs["drawstyle"],
                )

            if animated:
                animate, init = _set_animation(
                    pp_sampled_vals, ax, dtype=dtype, kind=kind, plot_kwargs=plot_kwargs
                )

            else:
                # run plot_kde manually with one plot call
                pp_densities = []
                for vals in pp_sampled_vals:
                    vals = np.array([vals]).flatten()
                    if dtype == "f":
                        pp_density, lower, upper = _fast_kde(vals)
                        pp_x = np.linspace(lower, upper, len(pp_density))
                        pp_densities.extend([pp_x, pp_density])
                    else:
                        nbins = round(len(vals) ** 0.5)
                        hist, bin_edges = np.histogram(vals, bins=nbins, density=True)
                        hist = np.concatenate((hist[:1], hist))
                        pp_densities.extend([bin_edges, hist])

                ax.plot(*pp_densities, **plot_kwargs)

            if mean:
                if dtype == "f":
                    plot_kde(
                        pp_vals.flatten(),
                        plot_kwargs={
                            "color": "C0",
                            "linestyle": "--",
                            "linewidth": linewidth,
                            "zorder": 2,
                        },
                        label="Posterior predictive mean {}".format(pp_var_name),
                        ax=ax,
                        legend=legend,
                    )
                else:
                    vals = pp_vals.flatten()
                    nbins = round(len(vals) ** 0.5)
                    hist, bin_edges = np.histogram(vals, bins=nbins, density=True)
                    hist = np.concatenate((hist[:1], hist))
                    ax.plot(
                        bin_edges,
                        hist,
                        color="C0",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name),
                        zorder=2,
                        linestyle="--",
                        drawstyle=plot_kwargs["drawstyle"],
                    )
            ax.tick_params(labelsize=xt_labelsize)
            ax.set_yticks([])

        elif kind == "cumulative":
            drawstyle = "default" if dtype == "f" else "steps-pre"
            ax.plot(
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
                    ax,
                    kind=kind,
                    alpha=alpha,
                    drawstyle=drawstyle,
                    linewidth=linewidth,
                )

            else:
                # run plot_kde manually with one plot call
                pp_densities = []
                for vals in pp_sampled_vals:
                    vals = np.array([vals]).flatten()
                    pp_x, pp_density = _empirical_cdf(vals)
                    pp_densities.extend([pp_x, pp_density])

                ax.plot(
                    *pp_densities, alpha=alpha, color="C5", drawstyle=drawstyle, linewidth=linewidth
                )
            ax.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))
            if mean:
                ax.plot(
                    *_empirical_cdf(pp_vals.flatten()),
                    color="C0",
                    linestyle="--",
                    linewidth=linewidth,
                    drawstyle=drawstyle,
                    label="Posterior predictive mean {}".format(pp_var_name)
                )
            ax.set_yticks([0, 0.5, 1])

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
                        ax=ax,
                        legend=legend,
                    )
                else:
                    vals = pp_vals.flatten()
                    nbins = round(len(vals) ** 0.5)
                    hist, bin_edges = np.histogram(vals, bins=nbins, density=True)
                    hist = np.concatenate((hist[:1], hist))
                    ax.plot(
                        bin_edges,
                        hist,
                        color="C0",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name),
                        zorder=3,
                        linestyle="--",
                        drawstyle="steps-pre",
                    )

            _, limit = ax.get_ylim()
            limit *= 1.05
            y_rows = np.linspace(0, limit, num_pp_samples + 1)
            jitter_scale = y_rows[1] - y_rows[0]
            scale_low = 0
            scale_high = jitter_scale * jitter

            obs_yvals = np.zeros_like(obs_vals, dtype=np.float64)
            if jitter:
                obs_yvals += np.random.uniform(low=scale_low, high=scale_high, size=len(obs_vals))
            ax.plot(
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
                    ax,
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
                    ax.plot(
                        vals, yvals, "o", zorder=2, color="C5", markersize=markersize, alpha=alpha
                    )

            ax.plot([], "C5o", label="Posterior predictive {}".format(pp_var_name))

            ax.set_yticks([])

        if var_name != pp_var_name:
            xlabel = "{} / {}".format(var_name, pp_var_name)
        else:
            xlabel = var_name
        ax.set_xlabel(make_label(xlabel, selection), fontsize=ax_labelsize)

        if legend:
            if i == 0:
                ax.legend(fontsize=xt_labelsize * 0.75)
            else:
                ax.legend([])

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
    if kind == "density":
        length = len(pp_sampled_vals)
        if dtype == "f":
            y_vals, lower, upper = _fast_kde(pp_sampled_vals[0])
            x_vals = np.linspace(lower, upper, len(y_vals))

            max_max = max([max(_fast_kde(pp_sampled_vals[i])[0]) for i in range(length)])

            ax.set_ylim(0, max_max)

            line, = ax.plot(x_vals, y_vals, **plot_kwargs)

            def animate(i):
                y_vals, lower, upper = _fast_kde(pp_sampled_vals[i])
                x_vals = np.linspace(lower, upper, len(y_vals))
                line.set_data(x_vals, y_vals)
                return line

        else:
            vals = pp_sampled_vals[0]
            y_vals, x_vals = np.histogram(vals, bins="auto", density=True)
            line, = ax.plot(x_vals[:-1], y_vals, **plot_kwargs)

            max_max = max(
                [
                    max(np.histogram(pp_sampled_vals[i], bins="auto", density=True)[0])
                    for i in range(length)
                ]
            )

            ax.set_ylim(0, max_max)

            def animate(i):
                y_vals, x_vals = np.histogram(pp_sampled_vals[i], bins="auto", density=True)
                line.set_data(x_vals[:-1], y_vals)
                return (line,)

    elif kind == "cumulative":
        x_vals, y_vals = _empirical_cdf(pp_sampled_vals[0])
        line, = ax.plot(
            x_vals, y_vals, alpha=alpha, color="C5", drawstyle=drawstyle, linewidth=linewidth
        )

        def animate(i):
            x_vals, y_vals = _empirical_cdf(pp_sampled_vals[i])
            line.set_data(x_vals, y_vals)
            return line

    elif kind == "scatter":
        x_vals = pp_sampled_vals[0]
        y_vals = np.full_like(x_vals, height, dtype=np.float64)
        line, = ax.plot(
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
