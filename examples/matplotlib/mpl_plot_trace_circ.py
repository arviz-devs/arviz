"""
Traceplot with Circular Variables
=================================
_gallery_category: Mixed Plots
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("glycan_torsion_angles")
az.plot_trace(data, var_names=["tors", "E"], circ_var_names=["tors"])

plt.show()
