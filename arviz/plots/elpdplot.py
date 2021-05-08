"""Plot pointwise elpd estimations of inference data."""
from copy import deepcopy
import numpy as np

from ..data import convert_to_inference_data
from ..rcparams import rcParams
from ..stats import ELPDData, loo, waic
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
    plot_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """
    Plot pointwise elpd differences between two or more models.

    Parameters
    ----------
    compare_dict : mapping, str -> ELPDData or InferenceData
        A dictionary mapping the model name to the object containing inference data or the result
        of `loo`/`waic` functions.
        Refer to az.convert_to_inference_data for details on possible dict items
    color : str or array_like, optional
        Colors of the scatter plot, if color is a str all dots will have the same color,
        if it is the size of the observations, each dot will have the specified color,
        otherwise, it will be interpreted as a list of the dims to be used for the color code
    xlabels : bool, optional
        Use coords as xticklabels
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on figsize.
    coords : mapping, optional
        Coordinates of points to plot. **All** values are used for computation, but only a
        a subset can be plotted for convenience.
    legend : bool, optional
        Include a legend to the plot. Only taken into account when color argument is a dim name.
    threshold : float
        If some elpd difference is larger than `threshold * elpd.std()`, show its label. If
        `None`, no observations will be highlighted.
    ic : str, optional
        Information Criterion (PSIS-LOO `loo`, WAIC `waic`) used to compare models. Defaults to
        ``rcParams["stats.information_criterion"]``.
        Only taken into account when input is InferenceData.
    scale : str, optional
        scale argument passed to az.loo or az.waic, see their docs for details. Only taken
        into account when input is InferenceData.
    plot_kwargs : dicts, optional
        Additional keywords passed to ax.scatter
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

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

    (compare_dict, scale, ic, ics) = Calculate_ICS(compare_dict, scale, ic)

    ic = ics[0]
        scales = [elpd_data["{}_scale".format(ic)] for elpd_data in compare_dict.values()]
        if not all(x == scales[0] for x in scales):
            raise SyntaxError(
                "All Information Criteria must be on the same scale, but {} are present".format(
                    set(scales)
                )
            )
    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    numvars = len(compare_dict)
    models = list(compare_dict.keys())

    if coords is None:
        coords = {}

    pointwise_data = [
        get_coords(compare_dict[model]["{}_i".format(ic)], coords) for model in models
    ]
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

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_elpd", "elpdplot", backend)
    ax = plot(**elpd_plot_kwargs)
    return ax

def Calculate_ICS(
    dataset_dict,
    scale: Optional[ScaleKeyword] = None,
    ic: Optional[ICKeyword] = None
):

"""
Helper Function used to execute the loo and waic information criteria

LOO is leave-one-out (PSIS-LOO `loo`) cross-validation and
WAIC is the widely applicable information criterion.
Read more theory here - in a paper by some of the leading authorities
on model selection dx.doi.org/10.1111/1467-9868.00353

Parameters
----------

dataset_dict :  dataset_dict: dict[str] -> InferenceData or ELPDData
    A dictionary of model names and InferenceData or ELPDData objects
scale: str, optional
        Output scale for IC. Available options are:

        - `log` : (default) log-score (after Vehtari et al. (2017))
        - `negative_log` : -1 * (log-score)
        - `deviance` : -2 * (log-score)

        A higher log-score (or a lower deviance) indicates a model with better predictive
        accuracy.
ic: str, optional
        Information Criterion (PSIS-LOO `loo` or WAIC `waic`) used to compare models. Defaults to
        ``rcParams["stats.information_criterion"]``.

Returns
-------

[WIP]

"""
    names = list(dataset_dict.keys())

    if scale is not None:
        scale = cast(ScaleKeyword, scale.lower())
    else:
        scale = cast(ScaleKeyword, rcParams["stats.ic_scale"])
    allowable = ["log", "negative_log", "deviance"] if NO_GET_ARGS else get_args(ScaleKeyword)
    if scale not in allowable:
        raise ValueError(f"{scale} is not a valid value for scale: must be in {allowable}")
    if scale == "log":
            scale_value = 1
            ascending = False
        else:
            if scale == "negative_log":
                scale_value = -1
            else:
                scale_value = -2
            ascending = True


    if ic is None:
        ic = cast(ICKeyword, rcParams["stats.information_criterion"])
    else:
        ic = cast(ICKeyword, ic.lower())
    allowable = ["loo", "waic"] if NO_GET_ARGS else get_args(ICKeyword)
    if ic not in allowable:
        raise ValueError(f"{ic} is not a valid value for ic: must be in {allowable}")


    # I have to return loo or waic in order for compare to create the df_comp and scale col


    if ic == "loo":
        ic_func: Callable = loo
    elif ic == "waic":
        ic_func = waic
    else:
        raise NotImplementedError(f"The information criterion {ic} is not supported.")
    

    names = []
    dataset_dict = deepcopy(dataset_dict)
    for name, dataset in dataset_dict.items():
        names.append(name)
        if not isinstance(dataset, ELPDData):
            try:
                dataset_dict[name] = ic_func(convert_to_inference_data(dataset), pointwise=True, scale=scale)
            except Exception as e:
                raise e.__class__(f"Encountered error trying to compute {ic} from model {name}.") from e    
    ics = [elpd_data.index[0] for elpd_data in compare_dict.values()]
    if not all(x == ics[0] for x in ics):
        raise SyntaxError(
            "All Information Criteria must be of the same kind, but both loo and waic data present"
        )
    return(dataset_dict, scale, ic, ics, name)
