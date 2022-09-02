"""
Density Plot
============
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")
az.plot_density(
    [centered_data, non_centered_data],
    data_labels=["Centered", "Non Centered"],
    var_names=["theta"],
    shade=0.1,
)
plt.show()
