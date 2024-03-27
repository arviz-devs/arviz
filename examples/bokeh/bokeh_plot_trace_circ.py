"""
Traceplot with Circular Variables
=================================
"""

import arviz as az

data = az.load_arviz_data("glycan_torsion_angles")
az.plot_trace(data, var_names=["tors", "E"], circ_var_names=["tors"], backend="bokeh")
