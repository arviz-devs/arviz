# Using NumPyro with ArviZ

ArviZ provides utilities to convert NumPyro sampling results into the
`InferenceData` format used for diagnostics and visualization.

This allows NumPyro users to analyze posterior samples using ArviZ's
plotting functions and statistical summaries.

## Minimal working example

The following example shows how to run a simple NumPyro model,
collect samples using MCMC, and convert the results to ArviZ
`InferenceData`.

```python
import arviz as az
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model():
    numpyro.sample("theta", dist.Normal(0, 1))

# run MCMC
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(random.PRNGKey(0))
```

# Using NumPyro with ArviZ

ArviZ provides utilities to convert NumPyro sampling results into the
`InferenceData` format used for diagnostics and visualization.

This allows NumPyro users to analyze posterior samples using ArviZ's
plotting functions and statistical summaries.

## Minimal working example

The following example shows how to run a simple NumPyro model,
collect samples using MCMC, and convert the results to ArviZ
`InferenceData`.

```python
import arviz as az
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def model():
    numpyro.sample("theta", dist.Normal(0, 1))

# run MCMC
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(random.PRNGKey(0))

# convert to ArviZ InferenceData
idata = az.from_numpyro(mcmc)
```

Inspecting the results
Once converted, the samples can be analyzed using ArviZ.

```python
# For example, to compute summary statistics:
az.summary(idata)

# To visualise thh sampling traces
az.plot_trace(idata)

# For posterior visualisation
az.plot_posterior(idata)
```

Why convert NumPyro results to ArviZ?

ArviZ provides a unified interface for analyzing Bayesian models.
By converting NumPyro outputs to InferenceData, users can take
advantage of ArviZ features such as:

posterior diagnostics

visualization tools

model comparison utilities

This makes it easier to analyze and compare models built with
different probabilistic programming libraries.


