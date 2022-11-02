"""Summary plot for model comparison."""
import numpy as np

from ..labels import BaseLabeller
from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_compare(
    comp_df,
    insample_dev=False,
    plot_standard_error=True,
    plot_ic_diff=True,
    order_by_rank=True,
    legend=True,
    title=True,
    figsize=None,
    textsize=None,
    labeller=None,
    plot_kwargs=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """
    Summary plot for model comparison.

    Models are compared based on their expected log pointwise predictive density (ELPD),
    the ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
    cross-validation (LOO) or using the widely applicable information criterion (WAIC).
    We recommend LOO in line with the work presented by Vehtari et al. (2016) available
    here: https://arxiv.org/abs/1507.04544.

    This plot is in the style of the one used in the book Statistical Rethinking by
    Richard McElreath.Chapter 6 in the first edition or 7 in the second.


    Parameters
    ----------
    comp_df : pd.DataFrame
        Result of the :func:`arviz.compare` method
    insample_dev : bool, optional
        Plot in-sample ELPD, that is the value of the information criteria without the
        penalization given by the effective number of parameters (p_loo or p_waic).
        Defaults to False
    plot_standard_error : bool, optional
        Plot the standard error of the ELPD. Defaults to True
    plot_ic_diff : bool, optional
        Plot standard error of the difference in ELPD between each model
        and the top-ranked model. Defaults to True
    order_by_rank : bool
        If True (default) ensure the best model is used as reference.
    legend : bool
        Add legend to figure. By default True.
    figsize : tuple, optional
        If None, size is (6, num of models) inches
    title : bool:
        Show a tittle with a description of how to interpret the plot. Defaults to True.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on ``figsize``.
    labeller : labeller instance, optional
        Class providing the method ``model_name_to_str`` to generate the labels in
        the plot.
        Read the :ref:`label_guide` for more details and usage examples.
    plot_kwargs : dict, optional
        Optional arguments for plot elements. Currently accepts 'color_ic',
        'marker_ic', 'color_insample_dev', 'marker_insample_dev', 'color_dse',
        'marker_dse', 'ls_min_ic' 'color_ls_min_ic',  'fontsize'
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
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
    plot_elpd : Plot pointwise elpd differences between two or more models.
    compare : Compare models based on PSIS-LOO loo or WAIC waic cross-validation.
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).
    waic : Compute the widely applicable information criterion.

    Examples
    --------
    Show default compare plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> model_compare = az.compare({'Centered 8 schools': az.load_arviz_data('centered_eight'),
        >>>                  'Non-centered 8 schools': az.load_arviz_data('non_centered_eight')})
        >>> az.plot_compare(model_compare)

    Include the in-sample ELDP

    .. plot::
        :context: close-figs

        >>> az.plot_compare(model_compare, insample_dev=True)

    """
    if plot_kwargs is None:
        plot_kwargs = {}

    if labeller is None:
        labeller = BaseLabeller()

    yticks_pos, step = np.linspace(0, -1, (comp_df.shape[0] * 2) - 1, retstep=True)
    yticks_pos[1::2] = yticks_pos[1::2] + step / 2
    labels = [labeller.model_name_to_str(model_name) for model_name in comp_df.index]

    if plot_ic_diff:
        yticks_labels = [""] * len(yticks_pos)
        yticks_labels[0] = labels[0]
        yticks_labels[2::2] = labels[1:]
    else:
        yticks_labels = labels

    _information_criterion = ["elpd_loo", "elpd_waic"]
    column_index = [c.lower() for c in comp_df.columns]
    for information_criterion in _information_criterion:
        if information_criterion in column_index:
            break
    else:
        raise ValueError(
            "comp_df must contain one of the following "
            f"information criterion: {_information_criterion}"
        )

    if order_by_rank:
        comp_df.sort_values(by="rank", inplace=True)

    compareplot_kwargs = dict(
        ax=ax,
        comp_df=comp_df,
        legend=legend,
        title=title,
        figsize=figsize,
        plot_ic_diff=plot_ic_diff,
        plot_standard_error=plot_standard_error,
        insample_dev=insample_dev,
        yticks_pos=yticks_pos,
        yticks_labels=yticks_labels,
        plot_kwargs=plot_kwargs,
        information_criterion=information_criterion,
        textsize=textsize,
        step=step,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_compare", "compareplot", backend)
    ax = plot(**compareplot_kwargs)

    return ax
