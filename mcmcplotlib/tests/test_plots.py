import matplotlib
matplotlib.use('Agg', warn=False)  # noqa
from pandas import DataFrame
import numpy as np
from ..plots import densityplot


def test_plots():
    trace = DataFrame({'a': np.random.rand(100)})

    densityplot(trace)
