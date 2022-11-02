# Plotting and reporting Bayes Factor given idata, var name, prior distribution and reference value
import logging

from scipy.stats import gaussian_kde

from ..data.utils import extract
from .plot_utils import get_plotting_function

_log = logging.getLogger(__name__)


def plot_bf(
    idata,
    var_name,
    prior=None,
    ref_val=0,
    xlim=None,
    colors=("C0", "C1"),
    figsize=None,
    textsize=None,
    hist_kwargs=None,
    plot_kwargs=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """
    Bayes Factor approximated as the Savage-Dickey density ratio.
    The Bayes factor is estimated by comparing a model
    against a model in which the parameter of interest has been restricted to a point-null.

    Parameters
    -----------
    idata : obj
        a :class:`arviz.InferenceData` object
    var_name : str
        Name of variable we want to test.
    prior : numpy.array, optional
        In case we want to use different prior, for example for sensitivity analysis.
    ref_val : int
        Point-null for Bayes factor estimation. Defaults to 0.
    xlim :  tuple, optional
        Set the x limits, which might be used for visualization purposes.
    colors : tuple
        Tuple of valid Matplotlib colors. First element for the prior, second for the posterior.
        Defaults to ('C0', 'C1').
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    plot_kwargs : dicts, optional
        Additional keywords passed to :func:`matplotlib.pyplot.plot`
    hist_kwargs : dicts, optional
        Additional keywords passed to :func:`arviz.plot_dist`. Only works for discrete variables
    ax : axes, optional
        :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.Figure`.
    backend : str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs : bool, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :func:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    dict : A dictionary with BF10 (Bayes Factor 10 (H1/H0 ratio), and BF01 (H0/H1 ratio).
    axes : matplotlib axes or bokeh figures

    Examples
    --------
    TBN

    """
    posterior = extract(idata, var_names=var_name)

    if ref_val > posterior.max() or ref_val < posterior.min():
        raise ValueError("Reference value is out of bounds of posterior")

    if posterior.ndim > 1:
        _log.info("Posterior distribution has {posterior.ndim} dimensions")

    if prior is None:
        prior = extract(idata, var_names=var_name, group="prior")

    if xlim is None:
        xlim = (prior.min(), prior.max())

    if posterior.dtype.kind == "f":
        posterior_pdf = gaussian_kde(posterior)
        prior_pdf = gaussian_kde(prior)

        posterior_at_ref_val = posterior_pdf(ref_val)
        prior_at_ref_val = prior_pdf(ref_val)

    elif posterior.dtype.kind == "i":
        prior_pdf = None
        posterior_pdf = None
        posterior_at_ref_val = (posterior == ref_val).mean()
        prior_at_ref_val = (prior == ref_val).mean()

    bf_10 = posterior_at_ref_val / prior_at_ref_val
    bf_01 = 1 / bf_10

    bfplot_kwargs = dict(
        ax=ax,
        bf_10=bf_10.item(),
        bf_01=bf_01.item(),
        xlim=xlim,
        prior=prior,
        posterior=posterior,
        prior_pdf=prior_pdf,
        posterior_pdf=posterior_pdf,
        ref_val=ref_val,
        prior_at_ref_val=prior_at_ref_val,
        posterior_at_ref_val=posterior_at_ref_val,
        var_name=var_name,
        colors=colors,
        figsize=figsize,
        textsize=textsize,
        hist_kwargs=hist_kwargs,
        plot_kwargs=plot_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    plot = get_plotting_function("plot_bf", "bfplot", backend)
    axes = plot(**bfplot_kwargs)
    return {"BF10": bf_10, "BF01": bf_01}, axes
