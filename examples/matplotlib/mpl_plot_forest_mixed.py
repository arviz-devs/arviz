"""
Forest Plot with ESS
====================
_gallery_category: Mixed Plots
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

centered_data = az.load_arviz_data("centered_eight")
ax = az.plot_forest_kwargs(
    centered_data,
    var_names=["theta"],
    figsize=(11.5, 5),
    colors="C1",
    ess=True,
    # r_hat=True,
    alternate_row_shading=False
)

plt.show()
