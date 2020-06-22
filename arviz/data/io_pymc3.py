"""PyMC3-specific conversion code."""
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Iterable, Union, TYPE_CHECKING
from types import ModuleType

import numpy as np
import xarray as xr
from .. import utils
from .inference_data import InferenceData, concat
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs, CoordSpec, DimSpec
from ..rcparams import rcParams

if TYPE_CHECKING:
    import pymc3 as pm
    from pymc3 import MultiTrace, Model  # pylint: disable=invalid-name
    import theano
    from typing import Set  # pylint: disable=ungrouped-imports
else:
    MultiTrace = Any  # pylint: disable=invalid-name
    Model = Any  # pylint: disable=invalid-name

___all__ = [""]

_log = logging.getLogger(__name__)

Coords = Dict[str, List[Any]]
Dims = Dict[str, List[str]]
# random variable object ...
Var = Any  # pylint: disable=invalid-name


def _monkey_patch_pymc3(pm: ModuleType) -> None:  # pylint: disable=invalid-name
    assert pm.__name__ == "pymc3"

    def fixed_eq(self, other):
        """Use object identity for MultiObservedRV equality."""
        return self is other

    if tuple([int(x) for x in pm.__version__.split(".")]) < (3, 9):  # type: ignore
        pm.model.MultiObservedRV.__eq__ = fixed_eq  # type: ignore


class PyMC3Converter:  # pylint: disable=too-many-instance-attributes
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
        log_likelihood=True,
        predictions=None,
        coords: Optional[Coords] = None,
        dims: Optional[Dims] = None,
        model=None,
        save_warmup: Optional[bool] = None,
    ):
        import pymc3
        import theano

        _monkey_patch_pymc3(pymc3)

        self.pymc3 = pymc3
        self.theano = theano

        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
        self.trace = trace

        # this permits us to get the model from command-line argument or from with model:
        try:
            self.model = self.pymc3.modelcontext(model or self.model)
        except TypeError:
            self.model = None

        if self.model is None:
            warnings.warn(
                "Using `from_pymc3` without the model will be deprecated in a future release. "
                "Not using the model will return less accurate and less useful results. "
                "Make sure you use the model argument or call from_pymc3 within a model context.",
                FutureWarning,
            )

        # This next line is brittle and may not work forever, but is a secret
        # way to access the model from the trace.
        self.attrs = None
        if trace is not None:
            if self.model is None:
                self.model = list(self.trace._straces.values())[  # pylint: disable=protected-access
                    0
                ].model
            self.nchains = trace.nchains if hasattr(trace, "nchains") else 1
            if hasattr(trace.report, "n_draws") and trace.report.n_draws is not None:
                self.ndraws = trace.report.n_draws
                self.attrs = {
                    "sampling_time": trace.report.t_sampling,
                    "tuning_steps": trace.report.n_tune,
                }
            else:
                self.ndraws = len(trace)
                if self.save_warmup:
                    warnings.warn(
                        "Warmup samples will be stored in posterior group and will not be"
                        " excluded from stats and diagnostics."
                        " Please consider using PyMC3>=3.9 and do not slice the trace manually.",
                        UserWarning,
                    )
            self.ntune = len(self.trace) - self.ndraws
            self.posterior_trace, self.warmup_trace = self.split_trace()
        else:
            self.nchains = self.ndraws = 0

        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.log_likelihood = log_likelihood
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
            elif posterior_predictive is not None:
                get_from = posterior_predictive
            elif prior is not None:
                get_from = prior
            if get_from is None:
                # pylint: disable=line-too-long
                raise ValueError(
                    "When constructing InferenceData must have at least"
                    " one of trace, prior, posterior_predictive or predictions."
                )

            aelem = arbitrary_element(get_from)
            self.ndraws = aelem.shape[0]

        self.coords = {} if coords is None else coords
        if hasattr(self.model, "coords"):
            self.coords = {**self.model.coords, **self.coords}

        self.dims = {} if dims is None else dims
        if hasattr(self.model, "RV_dims"):
            model_dims = {k: list(v) for k, v in self.model.RV_dims.items()}
            self.dims = {**model_dims, **self.dims}

        self.observations, self.multi_observations = self.find_observations()

    def find_observations(self) -> Tuple[Optional[Dict[str, Var]], Optional[Dict[str, Var]]]:
        """If there are observations available, return them as a dictionary."""
        if self.model is None:
            return (None, None)
        observations = {}
        multi_observations = {}
        for obs in self.model.observed_RVs:
            if hasattr(obs, "observations"):
                observations[obs.name] = obs.observations
            elif hasattr(obs, "data"):
                for key, val in obs.data.items():
                    multi_observations[key] = val.eval() if hasattr(val, "eval") else val
        return observations, multi_observations

    def split_trace(self) -> Tuple[Union[None, MultiTrace], Union[None, MultiTrace]]:
        """Split MultiTrace object into posterior and warmup.

        Returns
        -------
        trace_posterior: pymc3.MultiTrace or None
            The slice of the trace corresponding to the posterior. If the posterior
            trace is empty, None is returned
        trace_warmup: pymc3.MultiTrace or None
            The slice of the trace corresponding to the warmup. If the warmup trace is
            empty or ``save_warmup=False``, None is returned
        """
        trace_posterior = None
        trace_warmup = None
        if self.save_warmup and self.ntune > 0:
            trace_warmup = self.trace[: self.ntune]
        if self.ndraws > 0:
            trace_posterior = self.trace[self.ntune :]
        return trace_posterior, trace_warmup

    def log_likelihood_vals_point(self, point, var, log_like_fun):
        """Compute log likelihood for each observed point."""
        log_like_val = utils.one_de(log_like_fun(point))
        if var.missing_values:
            log_like_val = np.where(var.observations.mask, np.nan, log_like_val)
        return log_like_val

    @requires("trace")
    @requires("model")
    def _extract_log_likelihood(self, trace):
        """Compute log likelihood of each observation."""
        # If we have predictions, then we have a thinned trace which does not
        # support extracting a log likelihood.
        if self.log_likelihood is True:
            cached = [(var, var.logp_elemwise) for var in self.model.observed_RVs]
        else:
            cached = [
                (var, var.logp_elemwise)
                for var in self.model.observed_RVs
                if var.name in self.log_likelihood
            ]
        try:
            log_likelihood_dict = self.pymc3.sampling._DefaultTrace(  # pylint: disable=protected-access
                len(trace.chains)
            )
        except AttributeError:
            raise AttributeError(
                "Installed version of ArviZ requires PyMC3>=3.8. Please upgrade with "
                "`pip install pymc3>=3.8` or `conda install -c conda-forge pymc3>=3.8`."
            )
        for var, log_like_fun in cached:
            for chain in trace.chains:
                log_like_chain = [
                    self.log_likelihood_vals_point(point, var, log_like_fun)
                    for point in trace.points([chain])
                ]
                log_likelihood_dict.insert(var.name, np.stack(log_like_chain), chain)
        return log_likelihood_dict.trace_dict

    @requires("trace")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        var_names = self.pymc3.util.get_default_varnames(
            self.trace.varnames, include_transformed=False
        )
        data = {}
        data_warmup = {}
        for var_name in var_names:
            if self.warmup_trace:
                data_warmup[var_name] = np.array(
                    self.warmup_trace.get_values(var_name, combine=False, squeeze=False)
                )
            if self.posterior_trace:
                data[var_name] = np.array(
                    self.posterior_trace.get_values(var_name, combine=False, squeeze=False)
                )
        return (
            dict_to_dataset(
                data, library=self.pymc3, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
            dict_to_dataset(
                data_warmup,
                library=self.pymc3,
                coords=self.coords,
                dims=self.dims,
                attrs=self.attrs,
            ),
        )

    @requires("trace")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from PyMC3 trace."""
        data = {}
        rename_key = {"model_logp": "lp"}
        data = {}
        data_warmup = {}
        for stat in self.trace.stat_names:
            name = rename_key.get(stat, stat)
            if name == "tune":
                continue
            if self.warmup_trace:
                data_warmup[name] = np.array(
                    self.warmup_trace.get_sampler_stats(stat, combine=False)
                )
            if self.posterior_trace:
                data[name] = np.array(self.posterior_trace.get_sampler_stats(stat, combine=False))

        return (
            dict_to_dataset(
                data, library=self.pymc3, dims=None, coords=self.coords, attrs=self.attrs
            ),
            dict_to_dataset(
                data_warmup, library=self.pymc3, dims=None, coords=self.coords, attrs=self.attrs
            ),
        )

    @requires("trace")
    @requires("model")
    def log_likelihood_to_xarray(self):
        """Extract log likelihood and log_p data from PyMC3 trace."""
        if self.predictions or not self.log_likelihood:
            return None
        data_warmup = {}
        data = {}
        warn_msg = (
            "Could not compute log_likelihood, it will be omitted. "
            "Check your model object or set log_likelihood=False"
        )
        if self.posterior_trace:
            try:
                data = self._extract_log_likelihood(self.posterior_trace)
            except TypeError:
                warnings.warn(warn_msg)
        if self.warmup_trace:
            try:
                data_warmup = self._extract_log_likelihood(self.warmup_trace)
            except TypeError:
                warnings.warn(warn_msg)
        return (
            dict_to_dataset(data, library=self.pymc3, dims=self.dims, coords=self.coords),
            dict_to_dataset(data_warmup, library=self.pymc3, dims=self.dims, coords=self.coords),
        )

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
                    "This can mean that some draws or even whole chains are not represented.",
                    k,
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
        if self.observations is not None:
            prior_predictive_vars = list(self.observations.keys())
            prior_vars = [key for key in self.prior.keys() if key not in prior_predictive_vars]
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

    @requires(["observations", "multi_observations"])
    @requires("model")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        if self.predictions:
            return None
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        observed_data = {}
        for name, vals in {**self.observations, **self.multi_observations}.items():
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
        # For constant data, we are concerned only with deterministics and data.
        # The constant data vars must be either pm.Data (TensorSharedVariable) or pm.Deterministic
        constant_data_vars = {}  # type: Dict[str, Var]
        for var in self.model.deterministics:
            ancestors = self.theano.tensor.gof.graph.ancestors(var.owner.inputs)
            # no dependency on a random variable
            if not any((isinstance(a, self.pymc3.model.PyMC3Variable) for a in ancestors)):
                constant_data_vars[var.name] = var

        def is_data(name, var) -> bool:
            assert self.model is not None
            return (
                var not in self.model.deterministics
                and var not in self.model.observed_RVs
                and var not in self.model.free_RVs
                and var not in self.model.potentials
                and (self.observations is None or name not in self.observations)
            )

        # I don't know how to find pm.Data, except that they are named variables that aren't
        # observed or free RVs, nor are they deterministics, and then we eliminate observations.
        for name, var in self.model.named_vars.items():
            if is_data(name, var):
                constant_data_vars[name] = var

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
            # this might be a Deterministic, and must be evaluated
            elif hasattr(self.model[name], "eval"):
                vals = self.model[name].eval()
            vals = np.atleast_1d(vals)
            val_dims = dims.get(name)
            val_dims, coords = generate_dims_coords(
                vals.shape, name, dims=val_dims, coords=self.coords
            )
            # filter coords based on the dims
            coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in val_dims}
            try:
                constant_data[name] = xr.DataArray(vals, dims=val_dims, coords=coords)
            except ValueError as e:  # pylint: disable=invalid-name
                raise ValueError("Error translating constant_data variable %s: %s" % (name, e))
        return xr.Dataset(data_vars=constant_data, attrs=make_attrs(library=self.pymc3))

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (e.g., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        id_dict = {
            "posterior": self.posterior_to_xarray(),
            "sample_stats": self.sample_stats_to_xarray(),
            "log_likelihood": self.log_likelihood_to_xarray(),
            "posterior_predictive": self.posterior_predictive_to_xarray(),
            "predictions": self.predictions_to_xarray(),
            **self.priors_to_xarray(),
            "observed_data": self.observed_data_to_xarray(),
        }
        if self.predictions:
            id_dict["predictions_constant_data"] = self.constant_data_to_xarray()
        else:
            id_dict["constant_data"] = self.constant_data_to_xarray()
        return InferenceData(save_warmup=self.save_warmup, **id_dict)


def from_pymc3(
    trace: Optional[MultiTrace] = None,
    *,
    prior: Optional[Dict[str, Any]] = None,
    posterior_predictive: Optional[Dict[str, Any]] = None,
    log_likelihood: Union[bool, Iterable[str]] = True,
    coords: Optional[CoordSpec] = None,
    dims: Optional[DimSpec] = None,
    model: Optional[Model] = None,
    save_warmup: Optional[bool] = None,
) -> InferenceData:
    """Convert pymc3 data into an InferenceData object.

    All three of them are optional arguments, but at least one of ``trace``,
    ``prior`` and ``posterior_predictive`` must be present.
    For a usage example read the
    :doc:`Cookbook section on from_pymc3 </notebooks/InferenceDataCookbook>`

    Parameters
    ----------
    trace : pymc3.MultiTrace, optional
        Trace generated from MCMC sampling. Output of
        :py:func:`pymc3:pymc3.sampling.sample`.
    prior : dict, optional
        Dictionary with the variable names as keys, and values numpy arrays
        containing prior and prior predictive samples.
    posterior_predictive : dict, optional
        Dictionary with the variable names as keys, and values numpy arrays
        containing posterior predictive samples.
    log_likelihood : bool or array_like of str, optional
        List of variables to calculate `log_likelihood`. Defaults to True which calculates
        `log_likelihood` for all observed variables. If set to False, log_likelihood is skipped.
    coords : dict of {str: array-like}, optional
        Map of coordinate names to coordinate values
    dims : dict of {str: list of str}, optional
        Map of variable names to the coordinate names to use to index its dimensions.
    model : pymc3.Model, optional
        Model used to generate ``trace``. It is not necessary to pass ``model`` if in
        ``with`` context.
    save_warmup : bool, optional
        Save warmup iterations InferenceData object. If not defined, use default
        defined by the rcParams.

    Returns
    -------
    InferenceData
    """
    return PyMC3Converter(
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive,
        log_likelihood=log_likelihood,
        coords=coords,
        dims=dims,
        model=model,
        save_warmup=save_warmup,
    ).to_inference_data()


### Later I could have this return ``None`` if the ``idata_orig`` argument is supplied.  But
### perhaps we should have an inplace argument?
def from_pymc3_predictions(
    predictions,
    posterior_trace: Optional[MultiTrace] = None,
    model: Optional[Model] = None,
    coords=None,
    dims=None,
    idata_orig: Optional[InferenceData] = None,
    inplace: bool = False,
) -> InferenceData:
    """Translate out-of-sample predictions into ``InferenceData``.

    Parameters
    ----------
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
    idata_orig: InferenceData, optional
        If supplied, then modify this inference data in place, adding ``predictions`` and
        (if available) ``predictions_constant_data`` groups. If this is not supplied, make a
        fresh InferenceData
    inplace: boolean, optional
        If idata_orig is supplied and inplace is True, merge the predictions into idata_orig,
        rather than returning a fresh InferenceData object.

    Returns
    -------
    InferenceData:
        May be modified ``idata_orig``.
    """
    if inplace and not idata_orig:
        raise ValueError(
            (
                "Do not pass True for inplace unless passing"
                "an existing InferenceData as idata_orig"
            )
        )
    new_idata = PyMC3Converter(
        trace=posterior_trace, predictions=predictions, model=model, coords=coords, dims=dims
    ).to_inference_data()
    if idata_orig is None:
        return new_idata
    elif inplace:
        concat([idata_orig, new_idata], dim=None, inplace=True)
        return idata_orig
    else:
        # if we are not returning in place, then merge the old groups into the new inference
        # data and return that.
        concat([new_idata, idata_orig], dim=None, copy=True, inplace=True)
        return new_idata
