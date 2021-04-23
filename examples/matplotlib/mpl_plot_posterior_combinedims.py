import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate"]}
az.plot_posterior(data, var_names=["mu", "theta"], combine_dims=["school"], coords=coords, rope=(-1, 1))

plt.show()