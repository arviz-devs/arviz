"""
Bayes Factor Plot
=================
_gallery_category: Model Comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import arviz as az

idata = az.from_dict(
    posterior={"a": np.random.normal(1, 0.5, 5000)}, prior={"a": np.random.normal(0, 1, 5000)}
)

az.plot_bf(idata, var_name="a", ref_val=0)
plt.show()
