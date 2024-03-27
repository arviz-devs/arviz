"""
Matplotlib styles
=================
_gallery_category: Styles
_alt_text: Use Matplotlib Styles with `arviz.style.use()`.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import arviz as az

x = np.linspace(0, 1, 100)
dist = stats.beta(2, 5).pdf(x)

style_list = [
    "default",
    ["default", "arviz-colors"],
    "arviz-darkgrid",
    "arviz-whitegrid",
    "arviz-white",
    "arviz-grayscale",
    ["arviz-white", "arviz-redish"],
    ["arviz-white", "arviz-bluish"],
    ["arviz-white", "arviz-orangish"],
    ["arviz-white", "arviz-brownish"],
    ["arviz-white", "arviz-purplish"],
    ["arviz-white", "arviz-cyanish"],
    ["arviz-white", "arviz-greenish"],
    ["arviz-white", "arviz-royish"],
    ["arviz-white", "arviz-viridish"],
    ["arviz-white", "arviz-plasmish"],
    "arviz-doc",
    "arviz-docgrid",
]

fig = plt.figure(figsize=(20, 10), layout="constrained")
for idx, style in enumerate(style_list):
    with az.style.context(style, after_reset=True):
        ax = fig.add_subplot(5, 4, idx + 1, label=idx)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i in range(len(colors)):
            ax.plot(x, dist - i, f"C{i}", label=f"C{i}")
        ax.set_title(style)
        ax.set_ylabel("f(x)", rotation=0, labelpad=15)
        ax.set_xticklabels([])

plt.show()
