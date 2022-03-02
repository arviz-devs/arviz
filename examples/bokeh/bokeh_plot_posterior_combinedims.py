import arviz as az

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate", "Mt. Hermon", "Deerfield"]}
ax = az.plot_posterior(
    data,
    var_names=["mu", "theta"],
    combine_dims={"school"},
    coords=coords,
    rope=(-1, 1),
    backend="bokeh",
)
