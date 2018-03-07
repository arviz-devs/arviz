import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg', warn=False)  # noqa
from ..plots import densityplot


def test_plots():
    trace = pd.DataFrame({'a': np.random.rand(100)})

    densityplot(trace)
