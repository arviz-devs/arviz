# pylint: disable=wildcard-import,invalid-name,wrong-import-position
"""ArviZ is a library for exploratory analysis of Bayesian models."""
__version__ = "0.4.1"

import os
import logging
from matplotlib.pyplot import style
import importlib

# add ArviZ's styles to matplotlib's styles
arviz_style_path = os.path.join(os.path.dirname(__file__), "plots", "styles")
style.core.USER_LIBRARY_PATHS.append(arviz_style_path)
style.core.reload_library()

# Configure logging before importing arviz internals
_log = logging.getLogger("arviz")


def numba_check():
    numba = importlib.util.find_spec("numba")
    return numba is not None


_numba_flag = numba_check()


def disable_numba():
    global _numba_flag
    _numba_flag = False


def enable_numba():
    global _numba_flag
    _numba_flag = True


if not logging.root.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    _log.setLevel(logging.INFO)
    _log.addHandler(handler)

from .data import *
from .plots import *
from .stats import *
