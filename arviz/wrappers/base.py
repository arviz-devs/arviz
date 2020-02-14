"""Base class for sampling wrappers."""
import numpy as np


# from ..data import InferenceData
from ..stats import wrap_xarray_ufunc as _wrap_xarray_ufunc


class SamplingWrapper:
    """Class wrapping sampling routines for its usage via ArviZ.

    Using a common class, all inference backends can be supported in ArviZ. Hence, statistical
    functions requiring refitting like Leave Future Out or Simulation Based Calibration can be
    performed from ArviZ.

    For more info on wrappers see :ref:`wrappers_api`

    Parameters
    ----------
    model
        The model object used for sampling.
    idata_orig: InferenceData, optional
        Original InferenceData object.
    log_like_fun: callable, optional
        For simple cases where the pointwise log likelihood is a Python function, this
        function will be used to calculate the log likelihood. Otherwise,
        ``point_log_likelihood`` method must be implemented.
    sample_kwargs: dict, optional
        Sampling kwargs are stored as class attributes for their usage in the ``sample``
        method.
    idata_kwargs: dict, optional
        kwargs are stored as class attributes to be used in the ``get_inference_data`` method.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.
    """

    def __init__(
        self, model, idata_orig=None, log_like_fun=None, sample_kwargs=None, idata_kwargs=None
    ):
        self.model = model

        # if not isinstance(idata_orig, InferenceData) or idata_orig is not None:
        #     raise TypeError("idata_orig must be of InferenceData type or None")
        self.idata_orig = idata_orig

        if log_like_fun is None or callable(log_like_fun):
            self.log_like_fun = log_like_fun
        else:
            raise TypeError("log_like_fun must be a callable object or None")

        self.sample_kwargs = {} if sample_kwargs is None else sample_kwargs
        self.idata_kwargs = {} if idata_kwargs is None else idata_kwargs

    def sel_observations(self, idx):
        """Select a subset of the observations in idata_orig.

        **Not implemented**: This method must be implemented by the SamplingWrapper subclasses.
        It is documented here to show its format and call signature.

        Parameters
        ----------
        idx
            Indexes to separate from the rest of the observed data.

        Returns
        -------
        modified_observed_data
            Observed data whose index is *not* ``idx``
        excluded_observed_data
            Observed data whose index is ``idx``
        """
        raise NotImplementedError("sel_observations method must be implemented for each subclass")

    def sample(self, modified_observed_data):
        """Sample ``self.model`` on the ``modified_observed_data`` subset.

        **Not implemented**: This method must be implemented by the SamplingWrapper subclasses.
        It is documented here to show its format and call signature.

        Parameters
        ----------
        modified_observed_data
            Data to fit the model on.

        Returns
        -------
        fitted_model
            Result of the fit.
        """
        raise NotImplementedError("sample method must be implemented for each subclass")

    def get_inference_data(self, fitted_model):
        """Convert the ``fitted_model`` to an InferenceData object.

        **Not implemented**: This method must be implemented by the SamplingWrapper subclasses.
        It is documented here to show its format and call signature.

        Parameters
        ----------
        fitted_model
            Result of the current fit.

        Returns
        -------
        idata_current: InferenceData
            InferenceData object containing the samples in ``fitted_model``
        """
        raise NotImplementedError("get_inference_data method must be implemented for each subclass")

    def point_log_likelihood(self, observation, parameters):
        """Pointwise log likelihood function.

        Parameters
        ----------
        observation
            Pointwise observation on which to calculate the log likelihood
        parameters
            Parameters on which the log likelihood is conditioned.

        Returns
        -------
        point_log_likelihood: float
            Value of the log likelihood of ``observation`` given ``parameters``
            according to ``self.model``
        """
        if self.log_like_fun is None:
            raise NotImplementedError(
                "If log_like_fun is None, point_log_likelihood method must "
                "be implemented for each subclass"
            )
        return self.log_like_fun(observation, parameters)

    def log_likelihood__i(self, excluded_obs, idata__i):
        r"""Get the log likelilhood samples :math:`\log p_{post(-i)}(y_i)`.

        Calculate the log likelihood of the data contained in excluded_obs using the
        model fitted with this data excluded, the results of which are stored in ``idata__i``.

        Parameters
        ----------
        excluded_obs
            Observations for which to calculate their log likelihood
        idata__i: InferenceData
            Inference results of refitting the data excluding some observations.

        Returns
        -------
        log_likelihood: xr.Dataarray
            Log likelihood of ``excluded_obs`` evaluated at each of the posterior samples
            stored in ``idata__i``.
        """
        ndraws = idata__i.posterior.dims["draw"]
        nchains = idata__i.posterior.dims["chain"]
        log_like_idx = _wrap_xarray_ufunc(
            lambda pars: self.point_log_likelihood(excluded_obs, pars),
            idata__i.posterior.to_array(),
            func_kwargs={"out": np.empty((nchains, ndraws))},
            ufunc_kwargs={"n_dims": 1, "ravel": False},
            input_core_dims=[["variable"]],
        )
        return log_like_idx

    def _check_method_is_implemented(self, method, *args):
        """Check a given method is implemented."""
        try:
            getattr(self, method)(*args)
        except NotImplementedError:
            return False
        except:  # pylint: disable=bare-except
            return True
        return True

    def check_implemented_methods(self, methods):
        """Check that all methods listed are implemented.

        Not all functions that require refitting need to have all the methods implemented in
        order to work properly. This function shoulg be used before using the SamplingWrapper and
        its subclasses to get informative error messages.

        Parameters
        ----------
        methods: list
            Check all elements in methods are implemented.

        Returns
        -------
            List with all non implemented methods
        """
        supported_methods_1arg = (
            "sel_observations",
            "sample",
            "get_inference_data",
        )
        supported_methods_2args = (
            "point_log_likelihood",
            "log_likelihood__i",
        )
        supported_methods = [*supported_methods_1arg, *supported_methods_2args]
        bad_methods = [method for method in methods if method not in supported_methods]
        if bad_methods:
            raise ValueError(
                "Not all method(s) in {} supported. Supported methods in SamplingWrapper "
                "subclasses are:{}".format(bad_methods, supported_methods)
            )

        not_implemented = []
        for method in methods:
            if method in supported_methods_1arg:
                if self._check_method_is_implemented(method, 1):
                    continue
                else:
                    not_implemented.append(method)
            elif method in supported_methods_2args:
                if self._check_method_is_implemented(method, 1, 1):
                    continue
                else:
                    not_implemented.append(method)
        return not_implemented
