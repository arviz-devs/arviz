(wrapper_guide)=
# Sampling wrappers
Sampling wrappers allow ArviZ to call PPLs in order to perform a limited
subset of their capabilities and calculate stats and diagnostics that require
refitting the model on different data.

Their implementation is still experimental and may vary in the future. In fact
there are currently two possible approaches when creating sampling wrappers.
The first one delegates all calculations to the PPL
whereas the second one externalizes the computation of the pointwise log
likelihood to the user who is expected to write it with xarray/numpy.

```{toctree}
pystan2_refitting
pystan_refitting
pymc3_refitting
numpyro_refitting
pystan2_refitting_xr_lik
pymc3_refitting_xr_lik
numpyro_refitting_xr_lik
```
