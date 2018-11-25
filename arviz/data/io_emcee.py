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
    num_vars = sampler.chain.shape[-1]
    # Get emcee version 2 sampler args, else get emcee version 3
    num_args = len(sampler.args) if hasattr(sampler, "args") else len(sampler.log_prob_fn.args)

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

    def __init__(self, sampler, *_, var_names=None, arg_names=None, coords=None, dims=None):
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
            data[var_name] = self.sampler.chain[(..., idx)]
        return dict_to_dataset(data, library=self.emcee, coords=self.coords, dims=self.dims)

    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        data = {}
        for idx, var_name in enumerate(self.arg_names):
            # Get emcee version 2 sampler args, else get emcee version 3
            data[var_name] = (
                self.sampler.args[idx]
                if hasattr(self.sampler, "args")
                else self.sampler.log_prob_fn.args[idx]
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


def from_emcee(sampler, *, var_names=None, arg_names=None, coords=None, dims=None):
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
    """
    return EmceeConverter(
        sampler, var_names=var_names, arg_names=arg_names, coords=coords, dims=dims
    ).to_inference_data()
