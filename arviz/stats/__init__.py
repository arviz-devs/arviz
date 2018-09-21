# pylint: disable=wildcard-import
"""Statistical tests and diagnostics for ArviZ."""
from .stats import *
from .diagnostics import *


__all__ = ['bfmi', 'compare', 'hpd', 'loo', 'psislw', 'r2_score', 'summary', 'waic', 'effective_n',
           'gelman_rubin', 'geweke', 'autocorr']
