"""
Compare Plot
============
_gallery_category: Model Comparison
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

model_compare = az.compare(
    {
        "Centered 8 schools": az.load_arviz_data("centered_eight"),
        "Non-centered 8 schools": az.load_arviz_data("non_centered_eight"),
    }
)
az.plot_compare(model_compare, figsize=(11.5, 5))

plt.show()
