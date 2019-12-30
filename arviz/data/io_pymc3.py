"""PyMC3-specific conversion code."""
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Set

import numpy as np
import xarray as xr
from .. import utils
from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs

if TYPE_CHECKING:
    import pymc3 as pm

_log = logging.getLogger(__name__)

Coords = Dict[str, List[Any]]
Dims = Dict[str, List[str]]


class PyMC3Converter: # pylint: disable=too-many-instance-attributes
    """Encapsulate PyMC3 specific logic."""

    model = None  # type: Optional[pm.Model]
    nchains = None  # type: int
    ndraws = None  # type: int
    posterior_predictive = None  # Type: Optional[Dict[str, np.ndarray]]
    predictions = None  # Type: Optional[Dict[str, np.ndarray]]
    prior = None  # Type: Optional[Dict[str, np.ndarray]]

    def __init__(
        self,
        *,
        trace=None,
        prior=None,
        posterior_predictive=None,
        predictions=None,
        coords: Optional[Coords] = None,
        dims: Optional[Dims] = None,
        model=None
    ):
        import pymc3

        self.pymc3 = pymc3

        self.trace = trace

        # This next line is brittle and may not work forever, but is a secret
        # way to access the model from the trace.
        if trace is not None:
            self.model = self.trace._straces[0].model  # pylint: disable=protected-access
            self.nchains = trace.nchains if hasattr(trace, "nchains") else 1
            self.ndraws = len(trace)
        else:
            self.model = None
            self.nchains = self.ndraws = 0

        # this permits us to get the model from command-line argument or from with model:
        try:
            self.model = self.pymc3.modelcontext(model or self.model)
        except TypeError:
            self.model = None

        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions

        def arbitrary_element(dct: Dict[Any, np.ndarray]) -> np.ndarray:
            return next(iter(dct.values()))

        if trace is None:
            # if you have a posterior_predictive built with keep_dims,
            # you'll lose here, but there's nothing I can do about that.
            self.nchains = 1
            get_from = None
            if predictions is not None:
                get_from = predictions
            elif prior is not None:
                get_from = prior
            elif posterior_predictive is not None:
                get_from = posterior_predictive
            if get_from is None:
                # pylint: disable=line-too-long
                raise ValueError("""When constructing InferenceData must have at least
                                    one of trace, prior, posterior_predictive or predictions.""")

            aelem = arbitrary_element(get_from)
            self.ndraws = aelem.shape[0]

        self.coords = coords
        self.dims = dims
        self.observations = (
            None
            if self.trace is None
            else True
            if any(
                hasattr(obs, "observations")
                for obs in self.trace._straces[  # pylint: disable=protected-access
                    0
                ].model.observed_RVs
            )
            else None
        )
        if self.observations is not None:
            self.observations = {obs.name: obs.observations for obs in self.model.observed_RVs}

    @requires("trace")
    @requires("model")
    def _extract_log_likelihood(self):
        """Compute log likelihood of each observation.

        Return None if there is not exactly 1 observed random variable.
        """
        # If we have predictions, then we have a thinned trace which does not
        # support extracting a log likelihood.
        if len(self.model.observed_RVs) != 1 or self.predictions:
            return None, None
        else:
            if self.dims is not None:
                coord_name = self.dims.get(
                    "log_likelihood", self.dims.get(self.model.observed_RVs[0].name)
                )
            else:
                coord_name = None

        cached = [(var, var.logp_elemwise) for var in self.model.observed_RVs]

        def log_likelihood_vals_point(point):
            """Compute log likelihood for each observed point."""
            log_like_vals = []
            for var, log_like in cached:
                log_like_val = utils.one_de(log_like(point))
                if var.missing_values:
                    log_like_val = log_like_val[~var.observations.mask]
                log_like_vals.append(log_like_val)
            return np.concatenate(log_like_vals)

        chain_likelihoods = []
        for chain in self.trace.chains:
            log_like = [log_likelihood_vals_point(point) for point in self.trace.points([chain])]
            chain_likelihoods.append(np.stack(log_like))
        return np.stack(chain_likelihoods), coord_name

    @requires("trace")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        var_names = self.pymc3.util.get_default_varnames(  # pylint: disable=no-member
            self.trace.varnames, include_transformed=False
        )
        data = {}
        for var_name in var_names:
            data[var_name] = np.array(self.trace.get_values(var_name, combine=False, squeeze=False))
        return dict_to_dataset(data, library=self.pymc3, coords=self.coords, dims=self.dims)

    @requires("trace")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from PyMC3 trace."""
        rename_key = {"model_logp": "lp"}
        data = {}
        for stat in self.trace.stat_names:
            name = rename_key.get(stat, stat)
            data[name] = np.array(self.trace.get_sampler_stats(stat, combine=False))
        log_likelihood, dims = self._extract_log_likelihood()
        if log_likelihood is not None:
            data["log_likelihood"] = log_likelihood
            dims = {"log_likelihood": dims}
        else:
            dims = None

        return dict_to_dataset(data, library=self.pymc3, dims=dims, coords=self.coords)

    def translate_posterior_predictive_dict_to_xarray(self, dct) -> xr.Dataset:
        """Take Dict of variables to numpy ndarrays (samples) and translate into dataset."""
        data = {}
        for k, ary in dct.items():
            shape = ary.shape
            if shape[0] == self.nchains and shape[1] == self.ndraws:
                data[k] = ary
            elif shape[0] == self.nchains * self.ndraws:
                data[k] = ary.reshape((self.nchains, self.ndraws, *shape[1:]))
            else:
                data[k] = utils.expand_dims(ary)
                # pylint: disable=line-too-long
                _log.warning(
                    "posterior predictive variable %s's shape not compatible with number of chains and draws. "
                    "This can mean that some draws or even whole chains are not represented.", k
                )
        return dict_to_dataset(data, library=self.pymc3, coords=self.coords, dims=self.dims)


    @requires(["posterior_predictive"])
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        return self.translate_posterior_predictive_dict_to_xarray(self.posterior_predictive)

    @requires(["predictions"])
    def predictions_to_xarray(self):
        """Convert predictions (out of sample predictions) to xarray."""
        return self.translate_posterior_predictive_dict_to_xarray(self.predictions)

    def priors_to_xarray(self):
        """Convert prior samples (and if possible prior predictive too) to xarray."""
        if self.prior is None:
            return {"prior": None, "prior_predictive": None}
        if self.trace is not None:
            prior_vars = self.pymc3.util.get_default_varnames(  # pylint: disable=no-member
                self.trace.varnames, include_transformed=False
            )
            prior_predictive_vars = [key for key in self.prior.keys() if key not in prior_vars]
        else:
            prior_vars = list(self.prior.keys())
            prior_predictive_vars = None

        priors_dict = {}
        for group, var_names in zip(
            ("prior", "prior_predictive"), (prior_vars, prior_predictive_vars)
        ):
            priors_dict[group] = (
                None
                if var_names is None
                else dict_to_dataset(
                    {k: utils.expand_dims(self.prior[k]) for k in var_names},
                    library=self.pymc3,
                    coords=self.coords,
                    dims=self.dims,
                )
            )
        return priors_dict

    @requires("observations")
    @requires("model")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        observed_data = {}
        for name, vals in self.observations.items():
            if hasattr(vals, "get_value"):
                vals = vals.get_value()
            vals = utils.one_de(vals)
            val_dims = dims.get(name)
            val_dims, coords = generate_dims_coords(
                vals.shape, name, dims=val_dims, coords=self.coords
            )
            # filter coords based on the dims
            coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in val_dims}
            observed_data[name] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data, attrs=make_attrs(library=self.pymc3))

    @requires(["trace", "predictions"])
    @requires("model")
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        # if both predictions AND trace are supplied, then the trace should be the
        # *thinned* trace used in prediction and NOT the full posterior trace. Hence we
        # give precedence here to checking the predictions.
        model_vars = None             # type: Set[str]
        if self.predictions is not None:
            model_vars = set(self.pymc3.util.get_default_varnames(
                self.predictions.keys(),
                include_transformed=True
            ))
        else:
            model_vars = set()
        if self.trace is not None:
            model_vars = model_vars | \
                         set(self.pymc3.util.get_default_varnames(  # pylint: disable=no-member
                             self.trace.varnames, include_transformed=True
                         ))
        if self.observations is not None:
            # pylint: disable=line-too-long
            model_vars = model_vars | \
                         {obs.name for obs in self.observations.values() if hasattr(obs, "name")} | \
                         set(self.observations.keys())

        # this check is necessary in filtering constant variables because I found that some still
        # slipped through, notably the bounding transformed variables introduced by bounded and
        # truncated RVs in PyMC3.
        def untransformed_name(name: str) -> str:
            if self.pymc3.util.is_transformed_name(name):
                return self.pymc3.util.get_untransformed_name(name)
            return name

        constant_data_vars = {
            name: var for name, var in self.model.named_vars.items() \
            if untransformed_name(name) not in model_vars
        }
        if not constant_data_vars:
            return None
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        constant_data = {}
        for name, vals in constant_data_vars.items():
            if hasattr(vals, "get_value"):
                vals = vals.get_value()
            vals = np.atleast_1d(vals)
            val_dims = dims.get(name)
            val_dims, coords = generate_dims_coords(
                vals.shape, name, dims=val_dims, coords=self.coords
            )
            # filter coords based on the dims
            coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in val_dims}
            try:
                constant_data[name] = xr.DataArray(vals, dims=val_dims, coords=coords)
            except ValueError as e: # pylint: disable=invalid-name
                raise ValueError("Error translating constant_data variable %s: %s"%(name, e))
        return xr.Dataset(data_vars=constant_data, attrs=make_attrs(library=self.pymc3))

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "predictions": self.predictions_to_xarray(),
                **self.priors_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
                "constant_data": self.constant_data_to_xarray(),
            }
        )


def from_pymc3(
    trace=None, *, prior=None, posterior_predictive=None, predictions=None,
    coords=None, dims=None, model=None
):
    """Convert pymc3 data into an InferenceData object."""
    return PyMC3Converter(
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive,
        predictions=predictions,
        coords=coords,
        dims=dims,
        model=model,
    ).to_inference_data()

def predictions_from_pymc3(predictions, posterior_trace, model,
                           coords=None, dims=None) -> InferenceData:
    """Special version of ``from_pymc3`` forout-of-sample predictions.

    Parameters
    ~~~~~~~~~~
    predictions: Dict[str, np.ndarray]
        The predictions are the return value of ``pymc3.sample_posterior_predictive``,
        a dictionary of strings (variable names) to numpy ndarrays (draws).
    posterior_trace: pm.MultiTrace
        This should be a trace that has been thinned appropriately for
        ``pymc3.sample_posterior_predictive``. Specifically, any variable whose shape is
        a deterministic function of the shape of any predictor (explanatory, independent, etc.)
        variables must be *removed* from this trace.
    model: pymc3.Model
        This argument is *not* optional, unlike in conventional uses of ``from_pymc3``.
        The reason is that the posterior_trace argument is likely to supply an incorrect
        value of model.
    coords: Dict[str, array-like[Any]]
        Coordinates for the variables.  Map from coordinate names to coordinate values.
    dims: Dict[str, array-like[str]]
        Map from variable name to ordered set of coordinate names.

    Returns
    ~~~~~~~
    InferenceData
    """
    return from_pymc3(trace=posterior_trace, predictions=predictions, model=model,
                      coords=coords, dims=dims)
