"""
Traceplot circular variables
============================

_thumb: .1, .8
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("glycan_torsion_angles")
az.plot_trace(data, var_names=["tors", "E"], circ_var_names=["tors"])

plt.show()
