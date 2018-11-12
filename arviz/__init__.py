# pylint: disable=wildcard-import,invalid-name,wrong-import-position
"""ArviZ is a library for exploratory analysis of Bayesian models."""
__version__ = "0.2.1"

import logging
from matplotlib.pyplot import style

from .data import *
from .plots import *
from .stats import *

if not logging.root.handlers:
    logging.getLogger(__name__).addHandler(logging.NullHandler())
