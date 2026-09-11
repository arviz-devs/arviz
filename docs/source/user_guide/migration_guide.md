---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

(migration_guide)=
# ArviZ migration guide

ArviZ 1.0 introduces breaking changes: a new data structure (DataTree), new defaults, and some API renames. This page is a short checklist of what you need to migrate. For the full story—why things changed, new features, and examples—see {ref}`whats_new_1_0`.

ArviZ is now split into three libraries (`arviz-base`, `arviz-stats`, `arviz-plots`). Unless you know you only need one part, you can get all three by importing the main package as usual:

```python
import arviz as az
```

## Plot names and signatures

Some plot functions were renamed or split; others kept the same name but have updated arguments and defaults. Use the following as a quick reference; see {ref}`whats_new_1_0` for the full inventory and new/removed functions.

**Renamed or restructured:**

| ArviZ &lt;1 | ArviZ ≥1 |
|-------------|----------|
| `plot_bpv` | `plot_ppc_pit`, `plot_ppc_tstat` |
| `plot_dist_comparison` | `plot_prior_posterior` |
| `plot_ecdf` | `plot_dist`, `plot_ecdf_pit` |
| `plot_ess` | `plot_ess`, `plot_ess_evolution` |
| `plot_forest` | `plot_forest`, `plot_ridge` |
| `plot_ppc` | `plot_ppc_dist` |
| `plot_posterior`, `plot_density` | `plot_dist` |
| `plot_trace` | `plot_trace_dist`, `plot_trace_rank` |

**Same name, updated implementation/arguments:** `plot_autocorr`, `plot_bf`, `plot_compare`, `plot_energy`, `plot_khat`, `plot_lm`, `plot_loo_pit`, `plot_mcse`, `plot_pair`, `plot_parallel`, `plot_rank`.

**Removed:** Legacy `plot_dist`, `plot_kde` (functionality is in the new `plot_dist`), `plot_violin`. Some functions (e.g. `plot_elpd`, `plot_ppc_residuals`, `plot_ts`) are planned but not yet available.

## Plot return type and kwargs

1. **Return type:** Plot functions now return a {class}`~arviz_plots.PlotCollection` (or similar) instead of raw axes or figures. That gives consistent layout, faceting, and control over how plots are combined.

   **What's affected:** All `plot_*` functions. Code that used the return value (e.g. to get axes or save a figure).

   **What to do:** If you only display the plot, you often need no change (the backend still shows it). If you need the figure or axes, use the PlotCollection API (e.g. `.show()`, or the methods that expose figures). See {ref}`whats_new_1_0` and the plotting docs.

Example:

```python
dt = az.load_arviz_data("centered_eight")
pc = az.plot_rank(dt, var_names=["mu", "tau"])
pc.show()   # PlotCollection controls layout; use pc for figure/axes access
```


2. **Kwargs:** Backend-specific options are no longer passed via many separate `*_kwargs` arguments (e.g. `plot_kwargs`, `ax_kwargs`). They are now passed via `visuals`, `stats`, and `**pc_kwargs`.

   **What's affected:** Code that passed backend or layout options into plot functions using the old keyword arguments.

   **What to do:** Use the `visuals`, `stats`, and `**pc_kwargs` arguments instead. See {ref}`whats_new_1_0` and the plotting docs for the mapping and examples.

   Example — before (legacy) vs after (1.0):

   ```python
   # TODO: We need an example that shows before and after
   ```

## DataTree (replaces InferenceData)

**What changed and why:** `arviz.InferenceData` is gone. ArviZ now uses {class}`xarray.DataTree` for the same groups (`posterior`, `observed_data`, etc.). DataTree is an xarray structure, so I/O is more flexible (any format xarray supports works) and groups can be nested.

**What's affected:** Code that uses `InferenceData` objects, type checks for `InferenceData`, or calls `.extend`, `.map`, or `.groups`. Code that does `idata["posterior"]` and then uses Dataset-only methods on the result may need a small change.

**What to do:**

- **Constructing / loading:** Converters and I/O return a DataTree; use it like you used InferenceData. Variable-level access is unchanged: `dt["posterior"]["theta"]` is still a DataArray.
- **When you need a Dataset from a group:** Use `dt["posterior"].dataset` (view) or `dt["posterior"].to_dataset()` (mutable copy). If you only use `dt["group"]["var"]`, no change.
- **Method replacements:** Use the table below. For `.extend`, use {meth}`xarray.DataTree.update` (order is reversed for the default left-like merge). For `.map` over some groups, use `.match(...)` or `.filter(...)` with `.map_over_datasets(...)`, then `.update()` to merge back into the full tree. For a list of top-level group names, use `list(dt.children)`; `dt.groups` exists but returns path-style strings.

| Old (InferenceData) | New (DataTree) |
|---------------------|----------------|
| `idata.extend(idata_new)` | `idata_new.update(idata)` |
| `idata.extend(idata_new, how="right")` | `idata.update(idata_new)` |
| `idata.map(fn, groups="_predictive", filter_groups="like")` | `dt.match("*_predictive").map_over_datasets(fn)`, then merge with `.update()` as needed |
| `idata.groups` (list of names) | `list(dt.children)` |

Example:

```python
dt = az.load_arviz_data("centered_eight")

# dt["posterior"] is a DataTree; variable access unchanged
dt["posterior"]["mu"]

# Top-level group names (replaces idata.groups)
list(dt.children)

# Need a Dataset? Use .dataset or .to_dataset()
dt["posterior"].to_dataset()
```

## Credible intervals

Three things changed for credible intervals and related defaults:

1. **Interval type (`ci_kind`):** Default is now equal-tailed interval (ETI) instead of highest-density interval (HDI). Many functions accept `ci_kind` (e.g. {func}`~arviz.summary`, {func}`~arviz_plots.plot_forest`).
2. **Parameter name (`hdi_prob` → `ci_prob`):** Renamed for consistency; the old name is removed in 1.0. Replace `hdi_prob` with `ci_prob` in code and in any rcParam keys.
3. **Default probability (0.94 → 0.89):** The default interval probability is now 0.89.

To override any of these (or other defaults), use a per-call argument, set `az.rcParams` at the start of your code, or use a user or project config file. See **rcParams and defaults** at the end of this page.

Examples:

```python
dt = az.load_arviz_data("centered_eight")

# 1. New defaults (no args): ETI, 89% probability
az.summary(dt, var_names=["mu"], kind="stats")
```

```python
# 2. New args explicit: same result as above
az.summary(dt, var_names=["mu"], kind="stats", ci_kind="eti", ci_prob=0.89)
```

```python
# 3. Restore old behavior: HDI, 94% probability
az.summary(dt, var_names=["mu"], kind="stats", ci_kind="hdi", ci_prob=0.94)
```

## Model comparison

1. **WAIC removed:** We use PSIS-LOO-CV only; WAIC is no longer available. PSIS-LOO-CV is more robust, has better diagnostics, and we've added features around it (e.g. LOO-R2, predictive metrics).

   **What's affected:** Code that called `waic` or relied on WAIC for model comparison or weights.

   **What to do:** Switch to {func}`~arviz.loo` (and optionally {func}`~arviz.compare`). See {ref}`whats_new_1_0` for new LOO-based features.


2. **Compare default (`ic_compare_method`):** The method used to compute model weights when you call {func}`~arviz.compare` (and related behavior) is controlled by `ic_compare_method`. The default is now **stacking** (weights chosen to optimize predictive performance) instead of the previous default (e.g. pseudo-BMA).

   **What's affected:** Code that uses {func}`~arviz.compare` (or depends on the default method for combining/ranking models). Reported weights and rankings can change even if you didn't call a method explicitly.

   **What to do:** If you're fine with stacking, no change needed. To use a different weighting method, call `compare(..., ic_compare_method="pseudo-bma")` (or another supported value) or set the corresponding rcParam. See {ref}`whats_new_1_0` for details.

Examples:

```python
dt = az.load_arviz_data("centered_eight")
dt_2 = az.load_arviz_data("non_centered_eight")

# 1. New defaults: LOO (replaces WAIC), compare with stacking
az.loo(dt)
az.compare({"model A": az.loo(dt), "model B": az.loo(dt_2)})
```

```python
# 2. New call explicit: same as above
az.compare({"model A": az.loo(dt), "model B": az.loo(dt_2)}, ic_compare_method="stacking")
```

```python
# 3. Restore old behavior: pseudo-BMA weights (in 1.0 pass ic_compare_method="pseudo-bma")
az.compare({"model A": az.loo(dt), "model B": az.loo(dt_2)}, method="pseudo-bma")
```

## I/O (netcdf/zarr)

**What changed and why:** Existing netcdf/zarr files are unchanged and still valid. The only API change is where write lives: there are no top-level `to_netcdf`/`to_zarr`; you call them on the DataTree, so any format xarray supports is available without ArviZ adding wrappers.

**What's affected:** Code that called `idata.to_netcdf(...)` or `idata.to_zarr(...)` (or the old top-level write helpers). Reading is still {func}`arviz.from_netcdf` and {func}`arviz.from_zarr`; they now return a DataTree.

**What to do:** For reading, keep using `az.from_netcdf(...)` or `az.from_zarr(...)`. For writing, call methods on the tree: `dt.to_netcdf(...)` or `dt.to_zarr(...)` (e.g. `dt.to_netcdf("out.nc", engine="h5netcdf")`). No change to file format or to read paths. The main impact for many users will be ensuring the right dependencies are installed for the format they use.

Example:

```python
dt = az.load_arviz_data("centered_eight")

# Before (legacy): write on the object or top-level; from_netcdf returned InferenceData
# idata.to_netcdf("out.nc")   # or arviz.to_netcdf(idata, "out.nc")
# idata = az.from_netcdf("out.nc")

# After (1.0): write on the DataTree; from_netcdf returns a DataTree
dt.to_netcdf("out.nc", engine="h5netcdf")
az.from_netcdf("out.nc")
```

## Dimensions (dim and sample_dims)

The default sample dimensions are still `(chain, draw)`; most users need no change. In 0.x, only a few entry points (e.g. `hdi` via a kwarg, or some plots with `skip_dims`) could reduce over other dimensions. In 1.0 we use `dim` or `sample_dims` depending on the function. If you relied on custom dimensions, see {ref}`whats_new_1_0` for the full explanation and which functions use which argument.

## rcParams and defaults

You can control ArviZ behavior in three ways:

- **User-level or project templates:** Put options in an `arvizrc` file (e.g. in your home directory or project root). See the template for available keys.
- **`rcParams` at the start of your code:** Set `az.rcParams["group.key"] = value` once after importing ArviZ; it applies for the rest of the session.
- **Per-call arguments:** Pass the option (e.g. `ci_kind="hdi"`, `ci_prob=0.94`) to each function that supports it.

We do not consider changing an rcParam *default* to be a breaking change. Defaults may change in minor releases. If you care strongly about particular defaults, use a user-level template or set `rcParams` at the start of your script.

**Note:** The `hdi_prob` → `ci_prob` *rename* was a breaking change but was done in preparation; existing arvizrc templates already use `ci_prob`. The default method for model-comparison weights was changed to stacking a while ago, so that is unchanged in 1.0. For 0.x default values, see [arvizrc.template (v0.x)](https://github.com/arviz-devs/arviz/blob/v0.x/arvizrc.template).

---

For rationale, examples, and everything new in 1.0, see {ref}`whats_new_1_0`.
