import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("non_centered_eight")
az.plot_violin(data, var_names=["theta"], combine_dims=["school"])

plt.show()