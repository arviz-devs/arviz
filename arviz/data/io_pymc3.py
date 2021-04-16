# pylint: disable=unused-import
"""PyMC3-specific conversion code."""

try:
    from pymc3 import (
        to_inference_data as from_pymc3,
        predictions_to_inference_data as from_pymc3_predictions,
    )
except ImportError:
    from .io_pymc3_3x import from_pymc3, from_pymc3_predictions
