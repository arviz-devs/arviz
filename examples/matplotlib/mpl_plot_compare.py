"""
Compare Plot
============

_thumb: .5, .5
_example_title: Comparison plot
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")


model_compare = az.compare(
    {
        "Centered 8 schools": az.load_arviz_data("centered_eight"),
        "Non-centered 8 schools": az.load_arviz_data("non_centered_eight"),
    }
)
az.plot_compare(model_compare, figsize=(12, 4))

plt.show()
