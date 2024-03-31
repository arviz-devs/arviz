"""
Violinplot
==========
"""

import arviz as az

data = az.load_arviz_data("non_centered_eight")
ax = az.plot_violin(data, var_names=["mu", "tau"], backend="bokeh")
