# pylint: disable=unused-import
"""PyMC3-specific conversion code."""
import pkg_resources
import packaging

__all__ = ["from_pymc3", "from_pymc3_predictions"]

pymc3_version = pkg_resources.get_distribution("pymc3").version

if packaging.version.parse(pymc3_version) < packaging.version.parse("4.0"):
    from .io_pymc3_3x import from_pymc3, from_pymc3_predictions
else:

    def from_pymc3(
        trace=None,
        *,
        prior=None,
        posterior_predictive=None,
        log_likelihood=None,
        coords=None,
        dims=None,
        model=None,
        save_warmup=None,
        density_dist_obs=True,
    ):
        raise NotImplementedError(
            "The converter has been moved to PyMC3 codebase, use pymc3.to_inference_data"
        )

    def from_pymc3_predictions(
        predictions,
        posterior_trace=None,
        model=None,
        coords=None,
        dims=None,
        idata_orig=None,
        inplace = False,
    ):
        raise NotImplementedError(
            "The converter has been moved to PyMC3 codebase, "
            "use pymc3.to_inference_data_predictions"
        )
