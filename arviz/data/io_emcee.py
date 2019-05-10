"""emcee-specific conversion code."""
from .inference_data import InferenceData
from .base import dict_to_dataset


def _verify_names(sampler, var_names, arg_names):
    """Make sure var_names and arg_names are assigned reasonably.

    This is meant to run before loading emcee objects into InferenceData.
    In case var_names or arg_names is None, will provide defaults. If they are
    not None, it verifies there are the right number of them.

    Throws a ValueError in case validation fails.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Fitted emcee sampler
    var_names : list[str] or None
        Names for the emcee parameters
    arg_names : list[str] or None
        Names for the args/observations provided to emcee

    Returns
    -------
    list[str], list[str]
        Defaults for var_names and arg_names
    """
    # There are 3 possible cases: emcee2, emcee3 and sampler read from h5 file (emcee3 only)
    if hasattr(sampler, "args"):
        num_vars = sampler.chain.shape[-1]
        num_args = len(sampler.args)
    elif hasattr(sampler, "log_prob_fn"):
        num_vars = sampler.get_chain().shape[-1]
        num_args = len(sampler.log_prob_fn.args)
    else:
        num_vars = sampler.get_chain().shape[-1]
        num_args = 0  # emcee only stores the posterior samples

    if var_names is None:
        var_names = ["var_{}".format(idx) for idx in range(num_vars)]
    if arg_names is None:
        arg_names = ["arg_{}".format(idx) for idx in range(num_args)]

    if len(var_names) != num_vars:
        raise ValueError(
            "The sampler has {} variables, but only {} var_names were provided!".format(
                num_vars, len(var_names)
            )
        )

    if len(arg_names) != num_args:
        raise ValueError(
            "The sampler has {} args, but only {} arg_names were provided!".format(
                num_args, len(arg_names)
            )
        )
    return var_names, arg_names


class EmceeConverter:
    """Encapsulate emcee specific logic."""

    def __init__(self, *, sampler, var_names=None, arg_names=None, coords=None, dims=None):
        var_names, arg_names = _verify_names(sampler, var_names, arg_names)
        self.sampler = sampler
        self.var_names = var_names
        self.arg_names = arg_names
        self.coords = coords
        self.dims = dims
        import emcee

        self.emcee = emcee

    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = {}
        for idx, var_name in enumerate(self.var_names):
            # Use emcee3 syntax, else use emcee2
            data[var_name] = (
                self.sampler.get_chain()[(..., idx)].T
                if hasattr(self.sampler, "get_chain")
                else self.sampler.chain[(..., idx)]
            )
        return dict_to_dataset(data, library=self.emcee, coords=self.coords, dims=self.dims)

    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        data = {}
        for idx, var_name in enumerate(self.arg_names):
            # Use emcee3 syntax, else use emcee2
            data[var_name] = (
                self.sampler.log_prob_fn.args[idx]
                if hasattr(self.sampler, "log_prob_fn")
                else self.sampler.args[idx]
            )
        return dict_to_dataset(data, library=self.emcee, coords=self.coords, dims=self.dims)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


def from_emcee(sampler=None, *, var_names=None, arg_names=None, coords=None, dims=None):
    """Convert emcee data into an InferenceData object.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Fitted sampler from emcee.
    var_names : list[str] (Optional)
        A list of names for variables in the sampler
    arg_names : list[str] (Optional)
        A list of names for args in the sampler
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    dims : dict[str] -> list[str]
        Map variable names to their coordinates

    Returns
    -------
    InferenceData

    Examples
    --------
    Passing an ``emcee.EnsembleSampler`` object to ``az.from_emcee`` converts it
    to an InferenceData object. Start defining the model and running the sampler:

    .. plot::
        :context: close-figs

        >>> import emcee
        >>> import numpy as np
        >>> import arviz as az
        >>> J = 8
        >>> y_obs = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
        >>> sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
        >>> def log_prior_8school(theta):
        >>>     mu, tau, eta = theta[0], theta[1], theta[2:]
        >>>     # Half-cauchy prior, hwhm=25
        >>>     if tau < 0:
        >>>         return -np.inf
        >>>     prior_tau = -np.log(tau ** 2 + 25 ** 2)
        >>>     prior_mu = -(mu / 10) ** 2  # normal prior, loc=0, scale=10
        >>>     prior_eta = -np.sum(eta ** 2)  # normal prior, loc=0, scale=1
        >>>     return prior_mu + prior_tau + prior_eta
        >>> def log_likelihood_8school(theta, y, s):
        >>>     mu, tau, eta = theta[0], theta[1], theta[2:]
        >>>     return -((mu + tau * eta - y) / s) ** 2
        >>> def lnprob_8school(theta, y, s):
        >>>     prior = log_prior_8school(theta)
        >>>     like_vect = log_likelihood_8school(theta, y, s)
        >>>     like = np.sum(like_vect)
        >>>     return like + prior
        >>> nwalkers, draws = 50, 7000
        >>> ndim = J + 2
        >>> pos = np.random.normal(size=(nwalkers, ndim))
        >>> pos[:, 1] = np.absolute(pos[:, 1])
        >>> sampler = emcee.EnsembleSampler(
        >>>     nwalkers,
        >>>     ndim,
        >>>     lnprob_8school,
        >>>     args=(y_obs, sigma)
        >>> )
        >>> sampler.run_mcmc(pos, draws);

    And convert the sampler to an InferenceData object. As emcee does not store variable
    names, they must be passed to the converter in order to have them:

    .. plot::
        :context: close-figs

        >>> var_names = ['mu', 'tau']+['eta{}'.format(i) for i in range(J)]
        >>> emcee_data = az.from_emcee(sampler, var_names=var_names)

    From an InferenceData object, ArviZ's native data structure, the posterior plot
    of the first 3 variables can be done in one line:

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(emcee_data, var_names=var_names[:3])

    And the trace:

    .. plot::
        :context: close-figs

        >>> az.plot_trace(emcee_data, var_names=['mu'])

    Emcee is an Affine Invariant MCMC Ensemble Sampler, thus, its chains are **not**
    independent, which means that many ArviZ functions can not be used, at least directly.
    However, it is possible to combine emcee and ArviZ and use most of ArviZ
    functionalities. The first step is to modify the probability function to use the
    ``blobs`` and store the log_likelihood, then rerun the sampler using the new function:

    .. plot::
        :context: close-figs


        >>> def lnprob_8school_blobs(theta, y, s):
        >>>     prior = log_prior_8school(theta)
        >>>     like_vect = log_likelihood_8school(theta, y, s)
        >>>     like = np.sum(like_vect)
        >>>     return like + prior, like_vect
        >>> sampler_blobs = emcee.EnsembleSampler(
        >>>     nwalkers,
        >>>     ndim,
        >>>     lnprob_8school_blobs,
        >>>     args=(y_obs, sigma)
        >>> )
        >>> sampler_blobs.run_mcmc(pos, draws);

    ArviZ has no support for the ``blobs`` functionality yet, but a workaround can be
    created. First make sure that the dimensions are in the order
    ``(chain, draw, *shape)``. It may also be a good idea to apply a burn-in period
    and to thin the draw dimension (which due to the correlations between chains and
    consecutive draws, won't reduce the effective sample size if the value is small enough).
    Then convert the numpy arrays to InferenceData, in this case using ``az.from_dict``:

    .. plot::
        :context: close-figs

        >>> burnin, thin = 500, 10
        >>> blobs = np.swapaxes(np.array(sampler_blobs.blobs), 0, 1)[:, burnin::thin, :]
        >>> chain = sampler_blobs.chain[:, burnin::thin, :]
        >>> posterior_dict = {"mu": chain[:, :, 0], "tau": chain[:, :, 1], "eta": chain[:, :, 2:]}
        >>> stats_dict = {"log_likelihood": blobs}
        >>> emcee_data = az.from_dict(
        >>>     posterior=posterior_dict,
        >>>     sample_stats=stats_dict,
        >>>     coords={"school": range(8)},
        >>>     dims={"eta": ["school"], "log_likelihood": ["school"]}
        >>> )

    To calculate the effective sample size emcee's functions must be used. There are
    many changes in emcee's API from version 2 to 3, thus, the calculation is different
    depending on the version. In addition, in version 2, the autocorrelation time raises
    an error if the chain is not long enough.

    .. plot::
        :context: close-figs

        >>> if emcee.__version__[0] == '3':
        >>>     ess=(draws-burnin)/sampler.get_autocorr_time(quiet=True, discard=burnin, thin=thin)
        >>> else:
        >>>     # to avoid error while generating the docs, the ess value is hard coded, it
        >>>     # should be calculated with:
        >>>     # ess = chain.shape[1] / emcee.autocorr.integrated_time(chain)
        >>>     ess = (draws-burnin)/30
        >>> reff = np.mean(ess) / (nwalkers * chain.shape[1])

    This value can afterwards be used to estimate the leave-one-out cross-validation using
    Pareto smoothed importance sampling with ArviZ and plot the results:

    .. plot::
        :context: close-figs

        >>> loo_stats = az.loo(emcee_data, reff=reff, pointwise=True)
        >>> az.plot_khat(loo_stats.pareto_k)
    """
    return EmceeConverter(
        sampler=sampler, var_names=var_names, arg_names=arg_names, coords=coords, dims=dims
    ).to_inference_data()
