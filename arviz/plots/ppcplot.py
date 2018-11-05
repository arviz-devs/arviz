"""Posterior predictive plot."""
import numpy as np
from .kdeplot import plot_kde, _fast_kde
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    make_label,
    default_grid,
    _create_axes_grid,
    get_coords,
)
from ..utils import _var_names


def plot_ppc(data, var_names=None, coords=None, flatten=None,
             obs_plot_style="auto", pp_plot_style="auto", pp_flatten="auto",
             alpha=0.2, mean=True, figsize=None, textsize=None, data_pairs=None):
    """
    Plot for Posterior Predictive checks.

    Parameters
    ----------
    data : az.InferenceData object
        InferenceData object containing the observed and posterior
        predictive data.
    var_names : list
        List of variables to be plotted. Defaults to all variables in the
        model if None.
    coords : dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Passed to `Dataset.sel`. Dimensions without a mapping specified will
        include all coordinates for that dimension. Defaults to including all
        coordinates for all dimensions if None.
    flatten : list
        List of dimensions to flatten. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
    obs_plot_style: str
        The style to plot the observed data with. Must be in ('auto',
        'scatter', 'pdf', 'cdf', 'hist'). 'scatter' will plot a 1d
        scatter of the data. 'cdf' and 'pdf' will plot cumulative and kernel
        density estimates of the data respectively. 'hist' will plot a  histogram of the
        data. Defaults to 'auto' where 'scatter' is chosen if the number of
        observations is less than 30, otherwise 'pdf'.
    pp_plot_style: str
        The style to plot the posterior predictive data with. Must be in ('auto',
        'pdf', 'cdf'). 'cdf' and 'pdf' will plot cumulative and kernel
        density estimates of the data respectively. Defaults to 'auto' where 'cdf'
        is chosen if the obs_plot_style is also 'cdf', otherwise 'pdf'.
    pp_flatten: str
        Whether to flatten the posterior predictive density plot across samples.
        Must be in ('auto', 'individual', 'combined', 'both').
        If 'individual', it will plot a separate density for each individual
        posterior predictive sample. If 'combined', it will plot a single
        density for all of the posterior predictive data flattened across
        samples. If 'both', it will plot both the flattened and individual
        kdes. Defaults to 'auto' where 'combined' is chosen if the number of
        observations in each sample is less than 30, otherwise 'both'.
    alpha : float
        Opacity of posterior predictive density curves. Defaults to 0.2.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines.
        If None it will be autoscaled based on figsize.
    data_pairs : dict
        Dictionary containing relations between observed data and posterior
        predictive data.
        Dictionary structure:
            Key = data var_name

            Value = posterior predictive var_name
        Example: `data_pairs = {'y' : 'y_hat'}`

        If None, it will assume that the observed data and the posterior
        predictive data have the same variable name.

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

    """
    for group in ("posterior_predictive", "observed_data"):
        if not hasattr(data, group):
            raise TypeError(
                '`data` argument must have the group "{group}" for ppcplot'.format(group=group)
            )

    observed = data.observed_data
    posterior_predictive = data.posterior_predictive

    #if density_type.lower() not in ("pdf", "cdf"):
    #    raise TypeError("`kind` argument must be either `pdf` or `cdf`")

    if data_pairs is None:
        data_pairs = {}

    data_threshold = 30

    if var_names is None:
        var_names = observed.data_vars;
    var_names = _var_names(var_names)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]

    if flatten is None:
      flatten = list(observed.dims.keys())

    if coords is None:
        coords = {}


    new_coords = {}
    for key in coords.keys():
        vals = coords[key]
        new_coords[key] = np.where(np.in1d(observed[key], vals))[0]
    coords = new_coords

    obs_plotters = list(xarray_var_iter(observed.isel(coords),
                                        skip_dims=set(flatten),
                                        var_names=var_names,
                                        combined=True))
    pp_plotters = list(xarray_var_iter(posterior_predictive.isel(coords),
                                       var_names=pp_var_names,
                                       skip_dims=set(flatten),
                                       combined=True))
    length_plotters = len(obs_plotters)
    print(length_plotters)
    rows, cols = default_grid(length_plotters)


    # TODO: Validate new argument inputs
    assert(len(pp_plotters) == length_plotters)

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    _, axes = _create_axes_grid(length_plotters, rows, cols, figsize=figsize)
    print(axes.shape)

    for i, ax in enumerate(np.ravel(axes)):

        # plot the observed data
        var_name, selection, x = obs_plotters[i]
        dtype = observed[var_name].dtype.kind
        print(var_name)
        print(selection)
        print(x.shape)
        x = x.flatten()
        print(x.shape)
        num_obs = len(x)

        # Plot the observed data
        if obs_plot_style == "auto":
            if num_obs < data_threshold:
                kind = "scatter"
            else:
                kind = "pdf"
        else:
            kind = obs_plot_style

        ax.set_title(make_label(var_name, selection))
        if kind == "pdf":
            if dtype == "f":
                plot_kde(
                    x,
                    label="Observed {}".format(var_name),
                    plot_kwargs={"color": "k", "linewidth": linewidth, "zorder": 3},
                    fill_kwargs={"alpha": 0},
                    ax=ax,
                )
            else:
                nbins = round(len(x) ** 0.5)
                hist, bin_edges = np.histogram(x, bins=nbins, density=True)
                hist = np.concatenate((hist[:1], hist))
                ax.plot(
                    bin_edges,
                    hist,
                    label="Observed {}".format(var_name),
                    color="k",
                    linewidth=linewidth,
                    zorder=3,
                    drawstyle="steps-pre",
                )
        elif kind == "cdf":
            if dtype == "f":
                ax.plot(
                    *_empirical_cdf(x),
                    color="k",
                    linewidth=linewidth,
                    label="Observed {}".format(var_name),
                    zorder=3
                )
            else:
                ax.plot(
                    *_empirical_cdf(x),
                    color="k",
                    linewidth=linewidth,
                    label="Observed {}".format(var_name),
                    drawstyle="steps-pre",
                    zorder=3
                )
        elif kind == "scatter":
            ax.plot(x, np.zeros_like(x), 'o', markersize=markersize)
        elif kind == "hist":
            # TODO: histogram support
            print("Figure out arviz way to histogram. Appropriate to add in this PR?")
        else:
            # TODO: Handle invalid obs_plot_type
            print("Invalid observed data plot type: %s" % kind)

        # Plot posterior predictive data
        pp_var_name, selection, x = pp_plotters[i]
        print(pp_var_name)
        print(selection)
        print(x.shape)
        x = x.squeeze()
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], np.prod(x.shape[1:])))
        print(x.shape)

        if pp_flatten == "auto":
          if num_obs < data_threshold:
              flatten = 'combined'
          else:
              flatten = 'both'
        else:
            flatten = pp_flatten

        if pp_plot_style == "auto":
            if obs_plot_style == "cdf":
                kind = "cdf"
            else:
                kind = "pdf"

        if kind == "pdf":
            if flatten in ['individual', 'both']:
                # run plot_kde manually with one plot call
                pp_densities = []
                for vals in x:
                    if dtype == "f":
                        pp_density, lower, upper = _fast_kde(vals)
                        pp_x = np.linspace(lower, upper, len(pp_density))
                        pp_densities.extend([pp_x, pp_density])
                    else:
                        nbins = round(len(vals) ** 0.5)
                        hist, bin_edges = np.histogram(vals, bins=nbins, density=True)
                        hist = np.concatenate((hist[:1], hist))
                        pp_densities.extend([bin_edges, hist])
                plot_kwargs = {"color": "C5", "alpha": alpha, "linewidth": 0.5 * linewidth}
                if dtype == "i":
                    plot_kwargs["drawstyle"] = "steps-pre"
                ax.plot(*pp_densities, **plot_kwargs)
                ax.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))
            if flatten in ['combined', 'both']:
                if dtype == "f":
                    plot_kde(
                        x.flatten(),
                        plot_kwargs={
                            "color": "C0",
                            "linestyle": "--",
                            "linewidth": linewidth,
                            "zorder": 2,
                        },
                        label="Posterior predictive mean {}".format(pp_var_name),
                        ax=ax,
                    )
                else:
                    vals = x.flatten()
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
                        drawstyle="steps-pre",
                    )
            if var_name != pp_var_name:
                xlabel = "{} / {}".format(var_name, pp_var_name)
            else:
                xlabel = var_name
            ax.set_xlabel(xlabel, fontsize=ax_labelsize)
            ax.tick_params(labelsize=xt_labelsize)
            ax.set_yticks([])

        elif kind == "cdf":
            if flatten in ['individual', 'both']:
              # run plot_kde manually with one plot call
              pp_densities = []
              for vals in x:
                  pp_x, pp_density = _empirical_cdf(vals)
                  pp_densities.extend([pp_x, pp_density])
              if dtype == "f":
                  ax.plot(*pp_densities, alpha=alpha, color="C5", linewidth=linewidth)
              else:
                  ax.plot(
                      *pp_densities,
                      alpha=alpha,
                      color="C5",
                      drawstyle="steps-pre",
                      linewidth=linewidth
                  )
              ax.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))

            if flatten in ['combined', 'both']:
                if dtype == "f":
                    ax.plot(
                        *_empirical_cdf(x.flatten()),
                        color="C0",
                        linestyle="--",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name)
                    )
                else:
                    ax.plot(
                        *_empirical_cdf(x.flatten()),
                        color="C0",
                        linestyle="--",
                        linewidth=linewidth,
                        drawstyle="steps-pre",
                        label="Posterior predictive mean {}".format(pp_var_name)
                    )
            if var_name != pp_var_name:
                xlabel = "{} / {}".format(var_name, pp_var_name)
            else:
                xlabel = var_name
            ax.set_xlabel(var_name, fontsize=ax_labelsize)
            ax.set_yticks([0, 0.5, 1])
        ax.legend(fontsize=xt_labelsize)
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
