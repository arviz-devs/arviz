import arviz as az
import pymc3 as pm
import numpy as np
import ujson as json

with open("radon.json", "rb") as f:
    radon_data = json.load(f)

radon_data = {
    key: np.array(value) if isinstance(value, list) else value
    for key, value in radon_data.items()
}

coords = {
    "Level": ["Basement", "Floor"],
    "obs_id": np.arange(radon_data["y"].size),
    "County": radon_data["county_name"],
    "g_coef": ["intercept", "slope"],
}

with pm.Model(coords=coords):
    floor_idx = pm.Data("floor_idx", radon_data["x"], dims="obs_id")
    county_idx = pm.Data("county_idx", radon_data["county"], dims="obs_id")
    uranium = pm.Data("uranium", radon_data["u"], dims="County")

    # Hyperpriors:
    g = pm.Normal("g", mu=0.0, sigma=10.0, dims="g_coef")
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts uranium model:
    a = pm.Deterministic("a", g[0] + g[1] * uranium, dims="County")
    za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, dims="County")
    a_county = pm.Deterministic("a_county", a + za_county * sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=radon_data["y"], dims="obs_id")

    prior = pm.sample_prior_predictive(500)
    trace = pm.sample(
        1000, tune=500, target_accept=0.99, random_seed=117
    )
    post_pred = pm.sample_posterior_predictive(trace)
    idata = az.from_pymc3(trace, prior=prior, posterior_predictive=post_pred)
