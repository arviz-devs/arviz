import matplotlib.pyplot as plt
import arviz as az

data = az.load_arviz_data('rugby')
az.plot_dist_comparison(data, var_names=["defs"], coords={"team" : ["Italy"]}, combine_dims=[''], backend='bokeh')