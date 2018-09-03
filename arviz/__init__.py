# pylint: disable=wildcard-import,invalid-name,wrong-import-position
"""ArviZ is a library for exploratory analysis of Bayesian models."""
__version__ = '0.1.0'
from matplotlib.pyplot import style

from .inference_data import InferenceData
from .plots import *
from .stats import *
from .utils import *
