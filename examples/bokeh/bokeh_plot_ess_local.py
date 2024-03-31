"""
ESS Local Plot
==============
"""

import arviz as az

idata = az.load_arviz_data("non_centered_eight")

ax = az.plot_ess(idata, var_names=["mu"], kind="local", rug=True, backend="bokeh")
