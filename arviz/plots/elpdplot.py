"""Plot pointwise elpd estimations of inference data."""
import numpy as np

from ..rcparams import rcParams
from ..stats import _calculate_ics
from ..utils import get_coords
from .plot_utils import format_coords_as_labels, get_plotting_function


def plot_elpd(
    compare_dict,
    color="C0",
    xlabels=False,
    figsize=None,
    textsize=None,
    coords=None,
    legend=False,
    threshold=None,
    ax=None,
    ic=None,
    scale=None,
    var_name=None,
    plot_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """
    Plot pointwise elpd differences between two or more models.

    Parameters
    ----------
    compare_dict : mapping of {str : ELPDData or InferenceData}
        A dictionary mapping the model name to the object containing inference data or the result
        of :func:`arviz.loo` or :func:`arviz.waic` functions.
        Refer to :func:`arviz.convert_to_inference_data` for details on possible dict items.
    color : str or array_like, optional
        Colors of the scatter plot. If color is a str all dots will have the same color.
        If it is the size of the observations, each dot will have the specified color.
        Otherwise, it will be interpreted as a list of the dims to be used for the color code.
    xlabels : bool, optional
        Use coords as xticklabels. Defaults to False.
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars).
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on ``figsize``.
    coords : mapping, optional
        Coordinates of points to plot. **All** values are used for computation, but only a
        subset can be plotted for convenience.
    legend : bool, optional
        Include a legend to the plot. Only taken into account when color argument is a dim name.
    threshold : float
        If some elpd difference is larger than ``threshold * elpd.std()``, show its label. If
        `None`, no observations will be highlighted.
    ic : str, optional
        Information Criterion ("loo" for PSIS-LOO, "waic" for WAIC) used to compare models.
        Defaults to ``rcParams["stats.information_criterion"]``.
        Only taken into account when input is :class:`arviz.InferenceData`.
    scale : str, optional
        Scale argument passed to :func:`arviz.loo` or :func:`arviz.waic`, see their docs for
        details. Only taken into account when values in ``compare_dict`` are
        :class:`arviz.InferenceData`.
    var_name : str, optional
        Argument passed to to :func:`arviz.loo` or :func:`arviz.waic`, see their docs for
        details. Only taken into account when values in ``compare_dict`` are
        :class:`arviz.InferenceData`.
    plot_kwargs : dicts, optional
        Additional keywords passed to :meth:`matplotlib.axes.Axes.scatter`.
    ax: axes, optional
        :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.Figure`.
    backend: str, optional
        Select plotting backend {"matplotlib", "bokeh"}. Defaults to "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or
        :func:`bokeh.plotting.figure`.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

    See Also
    --------
    plot_compare : Summary plot for model comparison.

    Examples
    --------
    Compare pointwise PSIS-LOO for centered and non centered models of the 8-schools problem
    using matplotlib.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata1 = az.load_arviz_data("centered_eight")
        >>> idata2 = az.load_arviz_data("non_centered_eight")
        >>> az.plot_elpd(
        >>>     {"centered model": idata1, "non centered model": idata2},
        >>>     xlabels=True
        >>> )

    .. bokeh-plot::
        :source-position: above

        import arviz as az
        idata1 = az.load_arviz_data("centered_eight")
        idata2 = az.load_arviz_data("non_centered_eight")
        az.plot_elpd(
            {"centered model": idata1, "non centered model": idata2},
            backend="bokeh"
        )

    """
    try:
        (compare_dict, _, ic) = _calculate_ics(compare_dict, scale=scale, ic=ic, var_name=var_name)
    except Exception as e:
        raise e.__class__("Encountered error in ic computation of plot_elpd.") from e

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    numvars = len(compare_dict)
    models = list(compare_dict.keys())

    if coords is None:
        coords = {}

    pointwise_data = [get_coords(compare_dict[model][f"{ic}_i"], coords) for model in models]
    xdata = np.arange(pointwise_data[0].size)
    coord_labels = format_coords_as_labels(pointwise_data[0]) if xlabels else None

    if numvars < 2:
        raise Exception("Number of models to compare must be 2 or greater.")

    elpd_plot_kwargs = dict(
        ax=ax,
        models=models,
        pointwise_data=pointwise_data,
        numvars=numvars,
        figsize=figsize,
        textsize=textsize,
        plot_kwargs=plot_kwargs,
        xlabels=xlabels,
        coord_labels=coord_labels,
        xdata=xdata,
        threshold=threshold,
        legend=legend,
        color=color,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    plot = get_plotting_function("plot_elpd", "elpdplot", backend)
    ax = plot(**elpd_plot_kwargs)
    return ax
