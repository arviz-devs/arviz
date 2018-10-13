# pylint: disable=wildcard-import,invalid-name,wrong-import-position
"""ArviZ is a library for exploratory analysis of Bayesian models."""
__version__ = "0.2.1"
from matplotlib import get_backend as mpl_get_backend
from matplotlib.pyplot import style

from .data import *
from .plots import *
from .stats import *

import logging

_log = logging.getLogger('arviz')
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    _log.addHandler(handler)
    _log.info('matplotlib backend:' + mpl_get_backend())
