from pandas import DataFrame
import numpy as np
from ..plots import densityplot, traceplot, energyplot, posteriorplot


def test_plots():
    trace = DataFrame({'a': np.random.rand(100)})
    trace_energy = DataFrame({'energy': np.random.rand(100)})

    densityplot(trace)
    traceplot(trace)
    posteriorplot(trace)
    energyplot(trace_energy)
