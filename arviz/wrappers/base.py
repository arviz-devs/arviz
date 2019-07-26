from ..data import InferenceData


class SamplingWrapper:
    """Class wrapping sampling routines for its usage via ArviZ.

    Using a common class, all inference backends can be supported in ArviZ. Hence, statistical
    functions requiring refitting like Leave Future Out or Simulation Based Calibration can be
    performed from ArviZ.

    For more info on wrappers see :ref:`wrappers`

    Parameters
    ----------
    model
        The model object used for sampling.
    idata_orig: InferenceData, optional
        Original InferenceData object.
    log_like_fun: callable, optional
        For simple cases where the pointwise log likelihood is a Python function, this function will be used to
        calculate the log likelihood. Otherwise, ``point_log_likelihood`` method must be implemented.
    sample_kwargs: dict, optional
        Sampling kwargs are stored as class attributes for their usage in the ``sample`` method.
    idata_kwargs: dict, optional
        kwargs are stored as class attributes to be used in the ``get_inference_data`` method.
    """

    def __init__(self, model, idata_orig=None, log_like_fun=None, sample_kwargs=None, idata_kwargs=None):
        self.model = model

        if not isinstance(idata_orig, InferenceData) or idata_orig is not None:
            raise TypeError("idata_orig must be of InferenceData type or None")
        self.idata_orig = idata

        if log_like_fun is None or callable(log_like_fun):
            self.log_like_fun = log_like_fun
        else:
            raise TypeError("log_like_fun must be a callable object or None")

        self.sample_kwargs = {} if sample_kwargs is None else sample_kwargs
        self.idata_kwargs = {} if idata_kwargs is None else idata_kwargs

    def sel_observations(self, idx):
        """Select a subset of the observations in idata_orig.

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

        Parameters
        ----------
        fitted_model
            Result of the current fit.

        Returns
        -------
        idata_current: InferenceData
            InferenceData object containing the samples in ``fitted_model``"""
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
            Value of the log likelihood of ``observation`` given ``parameters`` according to ``self.model``"""
        if self.log_like_fun is None:
            raise NotImplementedError("If log_like_fun is None, point_log_likelihood method must "
                                      "be implemented for each subclass")
        else:
            return self.log_like_fun(observation, parameters)

    def _check_implemented_methods(self, methods):
        """Check that all methods listed are implemented.

        Not all functions that require refitting need to have all the methods implemented in
        order to work properly. This function shoulg be used before using the SamplingWrapper and
        its subclasses to get informative error messages.

        Parameters
        ----------
        methods: list
            Check all elements in methods are implemented.
        """
        supported_methods = ("sel_observations", "sample", "get_inference_data", "point_log_likelihood")
        bad_methods = [method for method in methods if method not in supported_methods]
        if bad_methods:
            raise ValueError(
                "Not all method(s) in {} supported. Supported methods in SamplingWrapper "
                "subclasses are:{}".format(bad_methods, supported_methods)

            )
        not_implemented = []
        for method in supported_methods[:-1]:
            if method in methods:
                try:
                    getattr(self, method)(1)
                except NotImplementedError:
                    not_implemented.append(method)
                except:
                    pass
        if "point_log_likelihood" in methods:
            try:
                self.point_log_likelihood(1, 1)
            except NotImplementedError:
                not_implemented.append("point_log_likelihood")
            except:
                pass
        return not_implemented
