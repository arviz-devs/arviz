"""
Energy Plot
===========
_gallery_category: Inference Diagnostics
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("centered_eight")
ax = az.plot_energy(data, fill_color=("C0", "C1"))

ax.set_title("Energy Plot")

plt.show()
