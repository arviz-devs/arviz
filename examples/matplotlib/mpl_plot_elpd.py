"""
ELPD Plot
=========
_gallery_category: Model Comparison
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

d1 = az.load_arviz_data("centered_eight")
d2 = az.load_arviz_data("non_centered_eight")

az.plot_elpd({"Centered eight": d1, "Non centered eight": d2}, xlabels=True)

plt.show()
