"""emcee-specific conversion code."""
import warnings
import xarray as xr
import numpy as np

from .. import utils
from .inference_data import InferenceData
from .base import dict_to_dataset, generate_dims_coords, make_attrs


def _verify_names(sampler, var_names, arg_names, slices):
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
    slices : list[seq] or None
        slices to select the variables (used for multidimensional variables)

    Returns
    -------
    list[str], list[str], list[seq]
        Defaults for var_names, arg_names and slices
    """
    # There are 3 possible cases: emcee2, emcee3 and sampler read from h5 file (emcee3 only)
    if hasattr(sampler, "args"):
        ndim = sampler.chain.shape[-1]
        num_args = len(sampler.args)
    elif hasattr(sampler, "log_prob_fn"):
        ndim = sampler.get_chain().shape[-1]
        num_args = len(sampler.log_prob_fn.args)
    else:
        ndim = sampler.get_chain().shape[-1]
        num_args = 0  # emcee only stores the posterior samples

    if slices is None:
        slices = utils.arange(ndim)
        num_vars = ndim
    else:
        num_vars = len(slices)
    indexs = utils.arange(ndim)
    slicing_try = np.concatenate([utils.one_de(indexs[idx]) for idx in slices])
    if len(set(slicing_try)) != ndim:
        warnings.warn(
            "Check slices: Not all parameters in chain captured. "
            "{} are present, and {} have been captured.".format(ndim, len(slicing_try)),
            SyntaxWarning,
        )
    if len(slicing_try) != len(set(slicing_try)):
        warnings.warn(
            "Overlapping slices. Check the index present: {}".format(slicing_try), SyntaxWarning
        )

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
    return var_names, arg_names, slices


class EmceeConverter:
    """Encapsulate emcee specific logic."""

    def __init__(
        self,
        sampler,
        var_names=None,
        slices=None,
        arg_names=None,
        arg_groups=None,
        blob_names=None,
        blob_groups=None,
        coords=None,
        dims=None,
    ):
        var_names, arg_names, slices = _verify_names(sampler, var_names, arg_names, slices)
        self.sampler = sampler
        self.var_names = var_names
        self.slices = slices
        self.arg_names = arg_names
        self.arg_groups = arg_groups
        self.blob_names = blob_names
        self.blob_groups = blob_groups
        self.coords = coords
        self.dims = dims
        import emcee

        self.emcee = emcee

    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = {}
        for idx, var_name in zip(self.slices, self.var_names):
            # Use emcee3 syntax, else use emcee2
            data[var_name] = (
                self.sampler.get_chain()[(..., idx)].swapaxes(0, 1)
                if hasattr(self.sampler, "get_chain")
                else self.sampler.chain[(..., idx)]
            )
        return dict_to_dataset(data, library=self.emcee, coords=self.coords, dims=self.dims)

    def args_to_xarray(self):
        """Convert emcee args to observed and constant_data xarray Datasets."""
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        if self.arg_groups is None:
            self.arg_groups = ["observed_data" for _ in self.arg_names]
        if len(self.arg_names) != len(self.arg_groups):
            raise ValueError(
                "arg_names and arg_groups must have the same length, or arg_groups be None"
            )
        arg_groups_set = set(self.arg_groups)
        bad_groups = [
            group for group in arg_groups_set if group not in ("observed_data", "constant_data")
        ]
        if bad_groups:
            raise SyntaxError(
                "all arg_groups values should be either 'observed_data' or 'constant_data' "
                ", not {}".format(bad_groups)
            )
        obs_const_dict = {group: {} for group in arg_groups_set}
        for idx, (arg_name, group) in enumerate(zip(self.arg_names, self.arg_groups)):
            # Use emcee3 syntax, else use emcee2
            arg_array = np.atleast_1d(
                self.sampler.log_prob_fn.args[idx]
                if hasattr(self.sampler, "log_prob_fn")
                else self.sampler.args[idx]
            )
            arg_dims = dims.get(arg_name)
            arg_dims, coords = generate_dims_coords(
                arg_array.shape, arg_name, dims=arg_dims, coords=self.coords
            )
            # filter coords based on the dims
            coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in arg_dims}
            obs_const_dict[group][arg_name] = xr.DataArray(arg_array, dims=arg_dims, coords=coords)
        for key, values in obs_const_dict.items():
            obs_const_dict[key] = xr.Dataset(data_vars=values, attrs=make_attrs(library=self.emcee))
        return obs_const_dict

    def blobs_to_dict(self):
        """Convert blobs to dictionary {groupname: xr.Dataset}."""
        # Omit blob conversion if blob_names is none.
        # I should return {} instead of None when avoided
        if self.blob_names is None:
            return {}
        elif self.blob_groups is None:
            self.blob_groups = ["sample_stats" for _ in self.blob_names]
        if len(self.blob_names) != len(self.blob_groups):
            raise ValueError(
                "blob_names and blob_groups must have the same length, or blob_groups be None"
            )
        if int(self.emcee.__version__[0]) >= 3:
            blobs = self.sampler.get_blobs()
        else:
            blobs = np.array(self.sampler.blobs)
        if blobs is None or blobs.size == 0:
            raise ValueError("No blobs in sampler, blob_names must be None")
        if len(blobs.shape) == 2:
            blobs = np.expand_dims(blobs, axis=-1)
        blobs = blobs.swapaxes(0, 2)
        nblobs, nwalkers, ndraws, *_ = blobs.shape
        if len(self.blob_names) != nblobs and len(self.blob_names) != 1:
            raise ValueError(
                "Incorrect number of blob names. Expected {}, found {}".format(
                    nblobs, len(self.blob_names)
                )
            )
        blob_groups_set = set(self.blob_groups)
        idata_groups = ("posterior", "observed_data", "constant_data")
        if np.any(np.isin(list(blob_groups_set), idata_groups)):
            raise SyntaxError(
                "{} groups should not come from blobs. Using them here would "
                "overwrite their actual values".format(idata_groups)
            )
        blob_dict = {group: {} for group in blob_groups_set}
        if len(self.blob_names) == 1:
            blob_dict[self.blob_groups[0]][self.blob_names[0]] = blobs.swapaxes(0, 2).swapaxes(0, 1)
        else:
            for i_blob, (name, group) in enumerate(zip(self.blob_names, self.blob_groups)):
                # for coherent blobs (all having the same dimensions) one line is enough
                blob = blobs[i_blob]
                # for blobs of different size, we get an array of arrays, which we convert
                # to an ndarray per blob_name
                if blob.dtype == object:
                    blob = blob.reshape(-1)
                    blob = np.stack(blob)
                    blob = blob.reshape((nwalkers, ndraws, -1))
                blob_dict[group][name] = np.squeeze(blob)
        for key, values in blob_dict.items():
            blob_dict[key] = dict_to_dataset(
                values, library=self.emcee, coords=self.coords, dims=self.dims
            )
        return blob_dict

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        blobs_dict = self.blobs_to_dict()
        obs_const_dict = self.args_to_xarray()
        return InferenceData(
            **{"posterior": self.posterior_to_xarray(), **obs_const_dict, **blobs_dict}
        )


def from_emcee(
    sampler=None,
    var_names=None,
    slices=None,
    arg_names=None,
    arg_groups=None,
    blob_names=None,
    blob_groups=None,
    coords=None,
    dims=None,
):
    """Convert emcee data into an InferenceData object.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Fitted sampler from emcee.
    var_names : list[str] (Optional)
        A list of names for variables in the sampler
    slices : list[array-like] (Optional)
        A list containing the indexes of each variable. Should only be used
        for multidimensional variables.
    arg_names : list[str] (Optional)
        A list of names for args in the sampler
    arg_groups : list of str, optional
        A list of the group names (either ``observed_data`` or ``constant_data``) where
        args in the sampler are stored. If None, all args will be stored in observed
        data group.
    blob_names : list[str] (Optional)
        A list of names for blobs in the sampler. When None,
        blobs are omitted, independently of them being present
        in the sampler or not.
    blob_groups : list[str] (Optional)
        A list of the groups where blob_names variables
        should be assigned respectively. If blob_names!=None
        and blob_groups is None, all variables are assigned
        to sample_stats group
    coords : dict[str] -> list[str] (Optional)
        Map of dimensions to coordinates
    dims : dict[str] -> list[str] (Optional)
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
        >>> nwalkers, draws = 50, 700
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
    names, they must be passed to the converter in order to have them. It can also be useful
    to perform a burn in cut to the MCMC samples (see :meth:`arviz.InferenceData.sel` for
    more details):

    .. plot::
        :context: close-figs

        >>> var_names = ['mu', 'tau']+['eta{}'.format(i) for i in range(J)]
        >>> emcee_data = az.from_emcee(sampler, var_names=var_names).sel(draw=slice(100, None))

    From an InferenceData object, ArviZ's native data structure, the posterior plot
    of the first 3 variables can be done in one line:

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(emcee_data, var_names=var_names[:3])

    This way of calling ``from_emcee`` stores each `eta` as a different variable, called
    `etai`, however, they are in fact different dimensions of the same variable. This can
    be seen in the likelihood and prior functions:
    ``mu, tau, eta = theta[0], theta[1], theta[2:]``. ArviZ has support for
    multidimensional variables, and there is a way to tell it how to split the variables
    like it was done in the likelihood and prior functions:

    .. plot::
        :context: close-figs

        >>> emcee_data = az.from_emcee(sampler, slices=[0, 1, slice(2, None)])

    After checking the default variable names, the trace of one dimension of eta can be
    plotted using ArviZ syntax:

    .. plot::
        :context: close-figs

        >>> az.plot_trace(emcee_data, var_names=["var_2"], coords={"var_2_dim_0": 4})

    Emcee does not store per-draw sample stats, however, it has a functionality called
    blobs that allows to store any variable on a per-draw basis. It can be used
    to store some sample_stats or even posterior_predictive data. The first step is to
    modify the probability function to use the ``blobs`` and store the log_likelihood,
    then rerun the sampler using the new function:

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

    Here, the argument blob_names is added with respect to the previous examples. As the
    group is not specified, it will go to sample_stats.

    .. plot::
        :context: close-figs

        >>> dims = {"eta": ["school"], "log_likelihood": ["school"]}
        >>> data = az.from_emcee(
        >>>     sampler_blobs,
        >>>     var_names = ["mu", "tau", "eta"],
        >>>     slices=[0, 1, slice(2,None)],
        >>>     blob_names=["log_likelihood"],
        >>>     dims=dims,
        >>>     coords={"school": range(8)}
        >>> )

    Or in the case of even more complicated blobs, each corresponding to a different
    group of the InferenceData object:

    .. plot::
        :context: close-figs

        >>> def lnprob_8school_blobs(theta, y, sigma):
        >>>     mu, tau, eta = theta[0], theta[1], theta[2:]
        >>>     prior = log_prior_8school(theta)
        >>>     like_vect = log_likelihood_8school(theta, y, sigma)
        >>>     like = np.sum(like_vect)
        >>>     return like + prior, (like_vect, np.random.normal((mu + tau * eta), sigma))
        >>> sampler_blobs = emcee.EnsembleSampler(
        >>>     nwalkers,
        >>>     ndim,
        >>>     lnprob_8school_blobs,
        >>>     args=(y_obs, sigma),
        >>> )
        >>> sampler_blobs.run_mcmc(pos, draws);
        >>> dims = {"eta": ["school"], "log_likelihood": ["school"], "y": ["school"]}
        >>> data = az.from_emcee(
        >>>     sampler_blobs,
        >>>     var_names = ["mu", "tau", "eta"],
        >>>     slices=[0, 1, slice(2,None)],
        >>>     arg_names=["y","sigma"],
        >>>     blob_names=["log_likelihood", "y"],
        >>>     blob_groups=["sample_stats", "posterior_predictive"],
        >>>     dims=dims,
        >>>     coords={"school": range(8)}
        >>> )

    This last version, which contains both observed data and posterior predictive can be
    used to plot posterior predictive checks:

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, var_names=["y"], alpha=0.3, num_pp_samples=50)

    """
    return EmceeConverter(
        sampler=sampler,
        var_names=var_names,
        slices=slices,
        arg_names=arg_names,
        arg_groups=arg_groups,
        blob_names=blob_names,
        blob_groups=blob_groups,
        coords=coords,
        dims=dims,
    ).to_inference_data()
