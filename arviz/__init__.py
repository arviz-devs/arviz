# pylint: disable=wildcard-import,invalid-name,wrong-import-position
"""ArviZ is a library for exploratory analysis of Bayesian models."""
__version__ = "0.2.1"

import logging
from matplotlib.pyplot import style

# Configure logging before importing arviz internals
_log = logging.getLogger(__name__)

if not logging.root.handlers:
    handler = logging.StreamHandler()
    _log.setLevel(logging.INFO)
    _log.addHandler(handler)

from .data import *
from .plots import *
from .stats import *
