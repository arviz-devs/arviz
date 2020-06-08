"""
ELPD Plot
=========

_thumb: .8, .8
"""
import arviz as az

d1 = az.load_arviz_data("centered_eight")
d2 = az.load_arviz_data("non_centered_eight")

ax = az.plot_elpd({"Centered eight": d1, "Non centered eight": d2}, backend="bokeh")
