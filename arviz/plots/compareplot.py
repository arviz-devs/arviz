"""Summary plot for model comparison."""
import numpy as np

from ..labels import BaseLabeller
from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_compare(
    comp_df,
    insample_dev=True,
    plot_standard_error=True,
    plot_ic_diff=True,
    order_by_rank=True,
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

    This plot is in the style of the one used in the book Statistical Rethinking (Chapter 6)
    by Richard McElreath.

    Notes
    -----
    Defaults to comparing Leave-one-out (psis-loo) if present in comp_df column,
    otherwise compares Widely Applicable Information Criterion (WAIC)


    Parameters
    ----------
    comp_df : pd.DataFrame
        Result of the `az.compare()` method
    insample_dev : bool, optional
        Plot in-sample deviance, that is the value of the information criteria without the
        penalization given by the effective number of parameters (pIC). Defaults to True
    plot_standard_error : bool, optional
        Plot the standard error of the information criteria estimate. Defaults to True
    plot_ic_diff : bool, optional
        Plot standard error of the difference in information criteria between each model
         and the top-ranked model. Defaults to True
    order_by_rank : bool
        If True (default) ensure the best model is used as reference.
    figsize : tuple, optional
        If None, size is (6, num of models) inches
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    labeller : labeller instance, optional
        Class providing the method `model_name_to_str` to generate the labels in the plot.
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
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures


    Examples
    --------
    Show default compare plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> model_compare = az.compare({'Centered 8 schools': az.load_arviz_data('centered_eight'),
        >>>                  'Non-centered 8 schools': az.load_arviz_data('non_centered_eight')})
        >>> az.plot_compare(model_compare)

    Plot standard error and information criteria difference only

    .. plot::
        :context: close-figs

        >>> az.plot_compare(model_compare, insample_dev=False)

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

    _information_criterion = ["loo", "waic"]
    column_index = [c.lower() for c in comp_df.columns]
    for information_criterion in _information_criterion:
        if information_criterion in column_index:
            break
    else:
        raise ValueError(
            "comp_df must contain one of the following"
            " information criterion: {}".format(_information_criterion)
        )

    if order_by_rank:
        comp_df.sort_values(by="rank", inplace=True)

    compareplot_kwargs = dict(
        ax=ax,
        comp_df=comp_df,
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
