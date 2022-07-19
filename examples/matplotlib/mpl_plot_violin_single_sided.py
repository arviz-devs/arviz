"""
Violin plot single sided
===========

_thumb: .2, .8
_example_title: Single sided violin plot
"""
import arviz as az

data = az.load_arviz_data("rugby")

labeller = az.labels.MapLabeller(var_name_map={"defs": "atts | defs"})
axs = az.plot_violin(data, var_names=["atts"], side="left", show=False)
az.plot_violin(data, var_names=["defs"], side="right", labeller=labeller, ax=axs, show=True)
