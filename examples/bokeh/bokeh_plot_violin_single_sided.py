"""
Single sided Violinplot
=======================

_thumb: .2, .8
"""

import arviz as az

data = az.load_arviz_data("rugby")
labeller = az.labels.MapLabeller(var_name_map={"defs": "atts | defs"})

p1 = az.plot_violin(
    data.posterior["atts"], side="left", backend="bokeh", show=False, labeller=labeller
)
p2 = az.plot_violin(
    data.posterior["defs"],
    side="right",
    ax=p1,
    backend="bokeh",
    shade_kwargs={"color": "lightsalmon"},
    labeller=labeller,
)
