"""Plot kde or histograms and values from MCMC samples."""
from .backends import check_bokeh_version


def plot_trace(
    data,
    var_names=None,
    coords=None,
    divergences="bottom",
    figsize=None,
    textsize=None,
    rug=False,
    lines=None,
    compact=False,
    combined=False,
    legend=False,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    hist_kwargs=None,
    trace_kwargs=None,
    backend=None,
    **kwargs
):
    """Plot distribution (histogram or kernel density estimates) and sampled values.

    If `divergences` data is available in `sample_stats`, will plot the location of divergences as
    dashed vertical lines.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : string, or list of strings
        One or more variables to be plotted.
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    divergences : {"bottom", "top", None, False}
        Plot location of divergences on the traceplots. Options are "bottom", "top", or False-y.
    figsize : figure size tuple
        If None, size is (12, variables * 2)
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize. Not implemented for bokeh backend.
    rug : bool
        If True adds a rugplot. Defaults to False. Ignored for 2D KDE.
        Only affects continuous variables.
    lines : tuple
        Tuple of (var_name, {'coord': selection}, [line, positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
    compact : bool
        Plot multidimensional variables in a single plot.
    combined : bool
        Flag for combining multiple chains into a single line. If False (default), chains will be
        plotted separately.
    legend : bool
        Add a legend to the figure with the chain color code.
    plot_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    fill_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    rug_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    hist_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects discrete variables.
    trace_kwargs : dict
        Extra keyword arguments passed to `plt.plot`
    backend : str {"matplotlib", "bokeh"}
        Select backend engine.

    Returns
    -------
    axes : matplotlib axes


    Examples
    --------
    Plot a subset variables

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('non_centered_eight')
        >>> coords = {'school': ['Choate', 'Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords)

    Show all dimensions of multidimensional variables in the same plot

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, compact=True)

    Combine all chains into one distribution

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, combined=True)


    Plot reference lines against distribution and trace

    .. plot::
        :context: close-figs

        >>> lines = (('theta_t',{'school': "Choate"}, [-1]),)
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, lines=lines)

    """
    if backend is None or backend.lower() in ("mpl", "matplotlib"):
        from .backends.matplotlib.mpl_traceplot import _plot_trace_mpl

        axes = _plot_trace_mpl(
            data,
            var_names=var_names,
            coords=coords,
            divergences=divergences,
            figsize=figsize,
            textsize=textsize,
            rug=rug,
            lines=lines,
            compact=compact,
            combined=combined,
            legend=legend,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            hist_kwargs=hist_kwargs,
            trace_kwargs=trace_kwargs,
        )
    elif backend.lower() == "bokeh":
        check_bokeh_version()
        from .backends.bokeh.bokeh_traceplot import _plot_trace_bokeh

        axes = _plot_trace_bokeh(
            data,
            var_names=var_names,
            coords=coords,
            divergences=divergences,
            figsize=figsize,
            rug=rug,
            lines=lines,
            compact=compact,
            combined=combined,
            legend=legend,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            hist_kwargs=hist_kwargs,
            trace_kwargs=trace_kwargs,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            'Backend {} not implemented. Use {{"matplotlib", "bokeh"}}'.format(backend)
        )
    return axes
