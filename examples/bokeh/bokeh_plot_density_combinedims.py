import arviz as az

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")
ax = az.plot_density(
    [centered_data, non_centered_data],
    data_labels=["Centered", "Non Centered"],
    var_names=["theta"],
    combine_dims=['school'],
    shade=0.1,
    backend="bokeh",
)