# Plotting and reporting Bayes Factor given idata, var name, prior distribution and reference value
import logging


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# 
_log = logging.getLogger(__name__)

def plot_bf(
    idata,
    var_name,
    prior,
    ref_val=0,
    xlim=None,
    ax=None):
   
    """Bayes Factor approximated as the Savage-Dickey density ratio.
    The Bayes factor is estimated by comparing a model 
    against a model in which the parameter of interest has been restricted to a point-null.
   
    Parameters
    -----------
    :idata: obj
        a :class:`arviz.InferenceData` object
    :var_name: str 
        Name of variable we want to test.
    :prior: numpy.array, optional
        In case we want to use diffent prior (for sensitivity analysis of BF), 
        we can define one and sent it to the function.
    :ref_val: int, default is 0
        Reference value for BF testing
    :xlim: numpy.array, optional
        limit the x axis (might be used for visualization porpuses sometimes)

    Returns
    -------
    Bayes Factor 10, BF01 and ax

    Examples
    --------
    TBN
    
    """
    var_name = _var_names(var_name, idata)
    post = extract(idata, var_names=var_name)
    if prior is None:
        # grab prior from the data in case it wasn't defined by the user
        prior = extract(idata, var_names=var_name, group="prior")
    if post.ndim > 1:
        _log.info("Posterior distribution has {post.ndim} dimensions")
    # generate vector
    if xlim is None:
        x = np.linspace(np.min(prior), np.max(prior),5000)
    else:
        x = np.linspace(xlim[0], xlim[1],5000)
    my_pdf = stats.gaussian_kde(post)
    prior_pdf = stats.gaussian_kde(prior)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        x, my_pdf(x), "--", lw=2.5, alpha=0.6, label="Posterior"
    )  # distribution function
    ax.plot(x, prior_pdf(x), "r-", lw=2.5, alpha=0.6, label="Prior")
    if ref_val>np.max(post) | ref_val<np.min(post):
        _log.warning('Reference value is out of bounds of posterior')
    else:
        posterior = my_pdf(ref_val) # this gives the pdf at ref_val
    prior = prior_pdf(ref_val)
    BF10 = posterior / prior
    BF01 = prior / posterior
    _log.info(f"the Bayes Factor 10 is {BF10}"
                f"the Bayes Factor 01 is {BF01}"
    )
    ax.plot(ref_val, posterior, "ko", lw=1.5)
    ax.plot(ref_val, prior, "ko", lw=1.5)
    ax.set_xlabel(var_name)
    ax.set_ylabel("Density")
    plt.legend()
    
    return {'BF10': BF10, 'BF01':BF01}, ax
# end