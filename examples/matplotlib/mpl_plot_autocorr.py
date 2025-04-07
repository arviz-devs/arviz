"""
Autocorrelation Plot
====================
_gallery_category: Inference Diagnostics
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("centered_eight")
az.plot_autocorr(data, var_names=("tau", "mu"))
az.plot_autocorr(
    data,
    var_names=["mu", "theta"],
    backend="matplotlib",
    coords={"school": ["Deerfield", "Hotchkiss"]},
)

plt.show()
