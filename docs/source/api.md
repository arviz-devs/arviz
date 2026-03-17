# API

ArviZ is structured in 3 separated libraries, but we expect most users to use them all
together. All the functions and classes that are available at the top level
of these 3 libraries is also available as `arviz.<object name>`.

The complete API is described at the following pages:

* {doc}`arviz-base API <arviz_base:api/index>`
* {doc}`arviz-stats API <arviz_stats:api/index>`
* {doc}`arviz-plots API <arviz_plots:api/index>`

## How to use the API effectively

Most users will interact with ArviZ through the top-level `arviz` namespace.

For example:

```python
import arviz as az

summary = az.summary(idata)
az.plot_trace(idata)
```
Even though ArviZ is internally structured into multiple submodules (arviz_base, arviz_stats, arviz_plots), users typically do not need to import from these directly.

This design helps keep the user interface simple while maintaining modularity internally.
