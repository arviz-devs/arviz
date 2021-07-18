"""
Regression Plot.

==========================================
_thumb: .6, .5
"""
import xarray as xr
import numpy as np
import arviz as az

data = az.load_arviz_data("regression1d")
x = xr.DataArray(np.linspace(0, 1, 100))
data.add_groups({"constant_data": {"x1": x}})
data.constant_data["x"] = x
data.posterior["y_model"] = (
    data.posterior["intercept"] + data.posterior["slope"] * data.constant_data["x"]
)

az.plot_lm(idata=data, y="y", x="x", y_model="y_model", backend="bokeh", figsize=(12, 6))
