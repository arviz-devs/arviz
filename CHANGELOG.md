# Change Log

## v0.x.x Unreleased

### New features

### Maintenance and fixes

### Deprecation

### Documentation

## v0.22.0 (2025 Jul 9)

### New features
- `plot_pair` now has more flexible support for `reference_values` ([2438](https://github.com/arviz-devs/arviz/pull/2438))
- Make `arviz.from_numpyro(..., dims=None)` automatically infer dims from the numpyro model based on its numpyro.plate structure

### Maintenance and fixes
- `reference_values` and `labeller` now work together in `plot_pair` ([2437](https://github.com/arviz-devs/arviz/issues/2437))
- Fix `plot_lm` for multidimensional data ([2408](https://github.com/arviz-devs/arviz/issues/2408))
- Add [`scipy-stubs`](https://github.com/scipy/scipy-stubs) as a development dependency ([2445](https://github.com/arviz-devs/arviz/pull/2445))
- Test compare dataframe stays consistent independently of input order ([2407](https://github.com/arviz-devs/arviz/pull/2407))
- Fix hdi_probs behaviour in 2d `plot_kde` ([2460](https://github.com/arviz-devs/arviz/pull/2460))

### Documentation
- Added documentation for `reference_values` ([2438](https://github.com/arviz-devs/arviz/pull/2438))
- Add migration guide page to help switch over to the new `arviz-xyz` libraries ([2459](https://github.com/arviz-devs/arviz/pull/2459))

## v0.21.0 (2025 Mar 06)

### New features

### Maintenance and fixes
- Make `arviz.data.generate_dims_coords` handle `dims` and `default_dims` consistently ([2395](https://github.com/arviz-devs/arviz/pull/2395))
- Only emit a warning for custom groups in `InferenceData` when explicitly requested ([2401](https://github.com/arviz-devs/arviz/pull/2401))
- Splits Bayes Factor computation out from `az.plot_bf` into `az.bayes_factor` ([2402](https://github.com/arviz-devs/arviz/issues/2402))
- Update `method="sd"` of `mcse` to not use normality assumption ([2167](https://github.com/arviz-devs/arviz/pull/2167))
- Add exception in `az.plot_hdi` for `x` of type `str` ([2413](https://github.com/arviz-devs/arviz/pull/2413))

### Documentation
- Add example of ECDF comparison plot to gallery ([2178](https://github.com/arviz-devs/arviz/pull/2178))
- Change Twitter to X, including the icon ([2418](https://github.com/arviz-devs/arviz/pull/2418))
- Update Bokeh link in Installation.rst ([2425](https://github.com/arviz-devs/arviz/pull/2425))
- Add missing periods to the ArviZ community page ([2426](https://github.com/arviz-devs/arviz/pull/2426))
- Fix missing docstring ([2430](https://github.com/arviz-devs/arviz/pull/2430))

## v0.20.0 (2024 Sep 28)

### New features
- Add optimized simultaneous ECDF confidence bands ([2368](https://github.com/arviz-devs/arviz/pull/2368))
- Add support for setting groups with `idata[group]` ([2374](https://github.com/arviz-devs/arviz/pull/2374))

### Maintenance and fixes
- Make `dm-tree` and optional dependency ([2379](https://github.com/arviz-devs/arviz/pull/2379))
- Fix bug in `psislw` modifying input inplace  ([2377](https://github.com/arviz-devs/arviz/pull/2377))
- Fix behaviour of two dimensional KDE plot with recent matplotlib releases ([2383](https://github.com/arviz-devs/arviz/pull/2383))
- Make defaults in `plot_compare` more intuitive ([2388](https://github.com/arviz-devs/arviz/pull/2388))

### Documentation
- Added extensions of virtual environments in [.gitignore](https://github.com/arviz-devs/arviz/blob/main/.gitignore) ([2371](https://github.com/arviz-devs/arviz/issues/2371))
- Fixed the issue in the [Contribution References Documentation](https://python.arviz.org/en/latest/contributing/index.html) ([2369](https://github.com/arviz-devs/arviz/issues/2369))
- Improve docstrings for `loo` and `waic`  ([2366](https://github.com/arviz-devs/arviz/pull/2366))

## v0.19.0 (2024 Jul 19)

### New features
-  Use revised Pareto k threshold ([2349](https://github.com/arviz-devs/arviz/pull/2349))
-   Added arguments `ci_prob`, `eval_points`, `rvs`, and `random_state` to `plot_ecdf` ([2316](https://github.com/arviz-devs/arviz/pull/2316))
-   Deprecated rcParam `stats.hdi_prob` and replaced with `stats.ci_prob` ([2316](https://github.com/arviz-devs/arviz/pull/2316))
- Expose features from [arviz-base](https://arviz-base.readthedocs.io), [arviz-stats](https://arviz-stats.readthedocs.io) and [arviz-plots](https://arviz-plots.readthedocs.io) as `arviz.preview`
  submodule ([2361](https://github.com/arviz-devs/arviz/pull/2361))

### Maintenance and fixes
- Ensure support with numpy 2.0 ([2321](https://github.com/arviz-devs/arviz/pull/2321))
- Update testing strategy to include an environment without optional dependencies and
  an environment with [scientific python nightlies](https://anaconda.org/scientific-python-nightly-wheels) ([2321](https://github.com/arviz-devs/arviz/pull/2321))
- Address bokeh related deprecations ([2362](https://github.com/arviz-devs/arviz/pull/2362))

- Fix legend overwriting issue in `plot_trace` ([2334](https://github.com/arviz-devs/arviz/pull/2334))

### Deprecation
-  Support for arrays and DataArrays in plot_khat has been deprecated. Only ELPDdata will be supported in the future ([2349](https://github.com/arviz-devs/arviz/pull/2349))
-   Removed arguments `values2`, `fpr`, `pointwise`, and `pit` in `plot_ecdf` ([2316](https://github.com/arviz-devs/arviz/pull/2316))

## v0.18.0 (2024 Apr 4)

### New features
- Add new example data `rugby_field` and update `rugby` example data ([2322](https://github.com/arviz-devs/arviz/pull/2322))
- Support for `pytree`s and robust to nested dictionaries. ([2291](https://github.com/arviz-devs/arviz/pull/2291))
- Add `.close` method to `InferenceData` ([2338](https://github.com/arviz-devs/arviz/pull/2338))


### Maintenance and fixes
- Fix deprecation warnings in multiple dependencies ([2329](https://github.com/arviz-devs/arviz/pull/2329),
  [2332](https://github.com/arviz-devs/arviz/pull/2332) and [2333](https://github.com/arviz-devs/arviz/pull/2333))

### Deprecation

-   Removed arguments `values2`, `fpr`, `pointwise`, `npoints`, and `pit` in `plot_ecdf` ([2316](https://github.com/arviz-devs/arviz/pull/2316))

### Documentation

## v0.17.1 (2024 Mar 13)

### Maintenance and fixes
- Fix deprecations introduced in latest pandas and xarray versions, and prepare for numpy 2.0 ones ([2315](https://github.com/arviz-devs/arviz/pull/2315)))
- Refactor ECDF code ([2311](https://github.com/arviz-devs/arviz/pull/2311))
- Fix `plot_forest` when Numba is installed ([2319](https://github.com/arviz-devs/arviz/pull/2319))

## v0.17.0 (2023 Dec 22)

### New features
- Add prior sensitivity diagnostic `psens` ([2093](https://github.com/arviz-devs/arviz/pull/2093))
- Add filter_vars functionality to `InfereceData.to_dataframe`method ([2277](https://github.com/arviz-devs/arviz/pull/2277))

### Maintenance and fixes

-   Update requirements: matplotlib>=3.5, pandas>=1.4.0, numpy>=1.22.0 ([2280](https://github.com/arviz-devs/arviz/pull/2280))
-   Fix behaviour of `plot_ppc` when dimension order isn't `chain, draw, ...` ([2283](https://github.com/arviz-devs/arviz/pull/2283))
-   Avoid repeating the variable name in `plot_ppc`, `plot_bpv`, `plot_loo_pit`... when repeated. ([2283](https://github.com/arviz-devs/arviz/pull/2283))
-   Add support for the latest CmdStanPy. ([2287](https://github.com/arviz-devs/arviz/pull/2287))
-   Fix import error on windows due to missing encoding argument ([2300](https://github.com/arviz-devs/arviz/pull/2300))
-   Add ``__delitem__`` method to InferenceData ([2292](https://github.com/arviz-devs/arviz/pull/2292))

### Documentation
- Improve the docstring of `psislw` ([2300](https://github.com/arviz-devs/arviz/pull/2300))
- Rerun the quickstart and working with InferenceData notebooks ([2300](https://github.com/arviz-devs/arviz/pull/2300))

-   Several fixes in `plot_ppc` docstring ([2283](https://github.com/arviz-devs/arviz/pull/2283))

## v0.16.1 (2023 Jul 18)

### Maintenance and fixes

-   Fix Numba deprecation errors and incorrect nopython usage ([2268](https://github.com/arviz-devs/arviz/pull/2268))

### Documentation

-   Rerun Numba notebook

## v0.16.0 (2023 Jul 13)

### New features

-   Add InferenceData<->DataTree conversion functions ([2253](https://github.com/arviz-devs/arviz/pull/2253))
-   Bayes Factor plot: Use arviz's kde instead of the one from scipy ([2237](https://github.com/arviz-devs/arviz/pull/2237))
-   InferenceData objects can now be appended to existing netCDF4 files and to specific groups within them ([2227](https://github.com/arviz-devs/arviz/pull/2227))
-   Added facade functions `az.to_zarr` and `az.from_zarr` ([2236](https://github.com/arviz-devs/arviz/pull/2236))

### Maintenance and fixes

-   Replace deprecated np.product with np.prod ([2249](https://github.com/arviz-devs/arviz/pull/2249))
-   Fix numba deprecation warning ([2246](https://github.com/arviz-devs/arviz/pull/2246))
-   Fixes for creating numpy object array ([2233](https://github.com/arviz-devs/arviz/pull/2233) and [2239](https://github.com/arviz-devs/arviz/pull/2239))
-   Adapt histograms generated by plot_dist to input dtype ([2247](https://github.com/arviz-devs/arviz/pull/2247))

## v0.15.1 (2023 Mar 06)

### New features

### Maintenance and fixes

-   Fix memory usage and improve efficiency in `from_emcee` ([2215](https://github.com/arviz-devs/arviz/pull/2215))
-   Lower pandas version needed ([2217](https://github.com/arviz-devs/arviz/pull/2217))

### Deprecation

### Documentation

-   Update documentation for various plots ([2208](https://github.com/arviz-devs/arviz/pull/2208))

## v0.15.0 (2023 Feb 19)

### New features

-   Adds Savage-Dickey density ratio plot for Bayes factor approximation. ([2037](https://github.com/arviz-devs/arviz/pull/2037), [2152](https://github.com/arviz-devs/arviz/pull/2152))
-   Add `CmdStanPySamplingWrapper` and `PyMCSamplingWrapper` classes ([2158](https://github.com/arviz-devs/arviz/pull/2158))
-   Changed dependency on netcdf4-python to h5netcdf ([2122](https://github.com/arviz-devs/arviz/pull/2122))

### Maintenance and fixes

-   Fix `reloo` outdated usage of `ELPDData` ([2158](https://github.com/arviz-devs/arviz/pull/2158))
-   plot_bpv smooth discrete data only when computing u_values ([2179](https://github.com/arviz-devs/arviz/pull/2179))
-   Fix bug when beanmachine objects lack some fields ([2154](https://github.com/arviz-devs/arviz/pull/2154))
-   Fix gap for `plot_trace` with option `kind="rank_bars"` ([2180](https://github.com/arviz-devs/arviz/pull/2180))
-   Fix `plot_lm` unsupported usage of `np.tile` ([2186](https://github.com/arviz-devs/arviz/pull/2186))
-   Update `_z_scale` to work with SciPy 1.10 ([2186](https://github.com/arviz-devs/arviz/pull/2186))
-   Fix bug in BaseLabeller when combining with with NoVarLabeller ([2200](https://github.com/arviz-devs/arviz/pull/2200))

### Deprecation

### Documentation

-   Add PyMC and CmdStanPy sampling wrapper examples ([2158](https://github.com/arviz-devs/arviz/pull/2158))
-   Fix docstring for plot_trace chain_prop and compact_prop parameters ([2176](https://github.com/arviz-devs/arviz/pull/2176))
-   Add video of contributing to ArviZ webinar in contributing guide ([2184](https://github.com/arviz-devs/arviz/pull/2184))

## v0.14.0 (2022 Nov 15)

### New features

-   Add `weight_predictions` function to allow generation of weighted predictions from two or more InfereceData with `posterior_predictive` groups and a set of weights ([2147](https://github.com/arviz-devs/arviz/pull/2147))
-   Add Savage-Dickey density ratio plot for Bayes factor approximation. ([2037](https://github.com/arviz-devs/arviz/pull/2037), [2152](https://github.com/arviz-devs/arviz/pull/2152
-   Adds rug plot for observed variables to `plot_ppc`. ([2161](https://github.com/arviz-devs/arviz/pull/2161))

### Maintenance and fixes

-   Fix dimension ordering for `plot_trace` with divergences ([2151](https://github.com/arviz-devs/arviz/pull/2151))

## v0.13.0 (2022 Oct 22)

### New features

-   Add `side` argument to `plot_violin` to allow single-sided violin plots ([1996](https://github.com/arviz-devs/arviz/pull/1996))
-   Added support for Bean Machine via the function `from_beanmachine`. ([2107](https://github.com/arviz-devs/arviz/pull/2107)
-   Add support for warmup samples in `from_pystan` for PyStan 3. ([2132](https://github.com/arviz-devs/arviz/pull/2132)

### Maintenance and fixes

-   Add exception in `az.plot_hdi` for `x` of type `np.datetime64` and `smooth=True` ([2016](https://github.com/arviz-devs/arviz/pull/2016))
-   Change `ax.plot` usage to `ax.scatter` in `plot_pair` ([1990](https://github.com/arviz-devs/arviz/pull/1990))
-   Example data has been moved to the [arviz_example_data](https://github.com/arviz-devs/arviz_example_data) repository and is now included using git subtree.
    ([2096](https://github.com/arviz-devs/arviz/pull/2096) and [2105](https://github.com/arviz-devs/arviz/pull/2105))
-   Bokeh kde contour plots started to use `contourpy` package ([2104](https://github.com/arviz-devs/arviz/pull/2104))
-   Update default Bokeh markers for rcparams ([2104](https://github.com/arviz-devs/arviz/pull/2104))
-   Correctly (re)order dimensions for `bfmi` and `plot_energy` ([2126](https://github.com/arviz-devs/arviz/pull/2126))
-   Fix bug with the dimension order dependency ([2103](https://github.com/arviz-devs/arviz/pull/2103))
-   Add testing module for labeller classes ([2095](https://github.com/arviz-devs/arviz/pull/2095))
-   Skip compression for object dtype while creating a netcdf file ([2129](https://github.com/arviz-devs/arviz/pull/2129))
-   Fix issue in dim generation when default dims are present in user inputed dims ([2138](https://github.com/arviz-devs/arviz/pull/2138))
-   Save InferenceData level attrs to netcdf and zarr ([2131](https://github.com/arviz-devs/arviz/pull/2131))
-   Update tests and docs for updated example data ([2137](https://github.com/arviz-devs/arviz/pull/2137))
-   Copy coords before modifying in ppcplot ([2160](https://github.com/arviz-devs/arviz/pull/2160))

### Deprecation

-   Removed `fill_last`, `contour` and `plot_kwargs` arguments from `plot_pair` function ([2085](https://github.com/arviz-devs/arviz/pull/2085))

### Documentation

-   Add translation overview to contributing guide ([2041](https://github.com/arviz-devs/arviz/pull/2041))
-   Improve how to release page ([2144](https://github.com/arviz-devs/arviz/pull/2144))

## v0.12.1 (2022 May 12)

### New features

-   Add `stat_focus` argument to `arviz.summary` ([1998](https://github.com/arviz-devs/arviz/pull/1998))

### Maintenance and fixes

-   `psislw` now smooths log-weights even when shape is lower than `1/3`([2011](https://github.com/arviz-devs/arviz/pull/2011))
-   Fixes `from_cmdstanpy`, handles parameter vectors of length 1 ([2023](https://github.com/arviz-devs/arviz/pull/2023))
-   Fix typo in `BaseLabeller` that broke `NoVarLabeller` ([2018](https://github.com/arviz-devs/arviz/pull/2018))

### Documentation

-   Adding plotting guides ([2025](https://github.com/arviz-devs/arviz/pull/2025))
-   Update links to use new domain ([2013](https://github.com/arviz-devs/arviz/pull/2013))

## v0.12.0 (2022 Mar 23)

### New features

-   Add new convenience function `arviz.extract_dataset` ([1725](https://github.com/arviz-devs/arviz/pull/1725))
-   Add `combine_dims` argument to several functions ([1676](https://github.com/arviz-devs/arviz/pull/1676))
-   [experimental] Enable dask chunking information to be passed to `InferenceData.from_netcdf` with regex support ([1749](https://github.com/arviz-devs/arviz/pull/1749))
-   Allow kwargs to customize appearance of the mean in `plot_lm`
-   Add dict option to `from_cmdstan` log_likelihood parameter (as in `from_pystan`)
-   Unify model comparison API. Both `plot_compare`, `plot_elpd` can now take dicts of InferenceData or ELPDData ([1690](https://github.com/arviz-devs/arviz/pull/1690))
-   Change default for rcParam `stats.ic_pointwise` to True ([1690](https://github.com/arviz-devs/arviz/pull/1690))
-   Add new plot type: plot_ecdf ([1753](https://github.com/arviz-devs/arviz/pull/1753))

### Maintenance and fixes

-   Drop Python 3.6 support ([1430](https://github.com/arviz-devs/arviz/pull/1430))
-   Bokeh 3 compatibility. ([1919](https://github.com/arviz-devs/arviz/pull/1919))
-   Remove manual setting of 2d KDE limits ([1939](https://github.com/arviz-devs/arviz/pull/1939))
-   Pin to bokeh<3 version ([1954](https://github.com/arviz-devs/arviz/pull/1954))
-   Fix legend labels in plot_ppc to reflect prior or posterior. ([1967](https://github.com/arviz-devs/arviz/pull/1967))
-   Change `DataFrame.append` to `pandas.concat` ([1973](https://github.com/arviz-devs/arviz/pull/1973))
-   Fix axis sharing behaviour in `plot_pair`. ([1985](https://github.com/arviz-devs/arviz/pull/1985))
-   Fix parameter duplication problem with PyStan ([1962](https://github.com/arviz-devs/arviz/pull/1962))
-   Temporarily disable pyjags tests ([1963](https://github.com/arviz-devs/arviz/pull/1963))
-   Fix tuple bug in coords ([1695](https://github.com/arviz-devs/arviz/pull/1695))
-   Fix extend 'right' join bug ([1718](https://github.com/arviz-devs/arviz/pull/1718))
-   Update attribute handling for InferenceData ([1357](https://github.com/arviz-devs/arviz/pull/1357))
-   Fix R2 implementation ([1666](https://github.com/arviz-devs/arviz/pull/1666))
-   Added warning message in `plot_dist_comparison()` in case subplots go over the limit ([1688](https://github.com/arviz-devs/arviz/pull/1688))
-   Fix coord value ignoring for default dims ([2001](https://github.com/arviz-devs/arviz/pull/2001))
-   Fixed plot_posterior with boolean data ([1707](https://github.com/arviz-devs/arviz/pull/1707))
-   Fix min_ess usage in plot_ess ([2002](https://github.com/arviz-devs/arviz/pull/2002))

### Deprecation

### Documentation

-   Fixed typo in `Forestplot` documentation
-   Restructured contributing section and added several new pages to help contributing to docs ([1903](https://github.com/arviz-devs/arviz/pull/1903))

## v0.11.4 (2021 Oct 3)

### Maintenance and fixes

-   Fix standard deviation code in density utils by replacing it with `np.std`. ([1833](https://github.com/arviz-devs/arviz/pull/1833))

## v0.11.3 (2021 Oct 1)

### New features

-   Change order of regularization in `psislw` ([1943](https://github.com/arviz-devs/arviz/pull/1943))
-   Added `labeller` argument to enable label customization in plots and summary ([1201](https://github.com/arviz-devs/arviz/pull/1201))
-   Added `arviz.labels` module with classes and utilities ([1201](https://github.com/arviz-devs/arviz/pull/1201) and [1605](https://github.com/arviz-devs/arviz/pull/1605))
-   Added probability estimate within ROPE in `plot_posterior` ([1570](https://github.com/arviz-devs/arviz/pull/1570))
-   Added `rope_color` and `ref_val_color` arguments to `plot_posterior` ([1570](https://github.com/arviz-devs/arviz/pull/1570))
-   Improved retrieving or pointwise log likelihood in `from_cmdstanpy`, `from_cmdstan` and `from_pystan` ([1579](https://github.com/arviz-devs/arviz/pull/1579) and [1599](https://github.com/arviz-devs/arviz/pull/1599))
-   Added interactive legend to bokeh `forestplot` ([1591](https://github.com/arviz-devs/arviz/pull/1591))
-   Added interactive legend to bokeh `ppcplot` ([1602](https://github.com/arviz-devs/arviz/pull/1602))
-   Add more helpful error message for HDF5 problems reading `InferenceData` from NetCDF ([1637](https://github.com/arviz-devs/arviz/pull/1637))
-   Added `data.log_likelihood`, `stats.ic_compare_method` and `plot.density_kind` to `rcParams` ([1611](https://github.com/arviz-devs/arviz/pull/1611))
-   Improve error messages in `stats.compare()`, and `var_name` parameter. ([1616](https://github.com/arviz-devs/arviz/pull/1616))
-   Added ability to plot HDI contours to `plot_kde` with the new `hdi_probs` parameter. ([1665](https://github.com/arviz-devs/arviz/pull/1665))
-   Add dtype parsing and setting in all Stan converters ([1632](https://github.com/arviz-devs/arviz/pull/1632))
-   Add option to specify colors for each element in ppc_plot ([1769](https://github.com/arviz-devs/arviz/pull/1769))

### Maintenance and fixes

-   Fix conversion for numpyro models with ImproperUniform latent sites ([1713](https://github.com/arviz-devs/arviz/pull/1713))
-   Fixed conversion of Pyro output fit using GPUs ([1659](https://github.com/arviz-devs/arviz/pull/1659))
-   Enforced using coordinate values as default labels ([1201](https://github.com/arviz-devs/arviz/pull/1201))
-   Integrate `index_origin` with all the library ([1201](https://github.com/arviz-devs/arviz/pull/1201))
-   Fix pareto k threshold typo in reloo function ([1580](https://github.com/arviz-devs/arviz/pull/1580))
-   Preserve shape from Stan code in `from_cmdstanpy` ([1579](https://github.com/arviz-devs/arviz/pull/1579))
-   Updated `from_pystan` converters to follow schema convention ([1585](https://github.com/arviz-devs/arviz/pull/1585)
-   Used generator instead of list wherever possible ([1588](https://github.com/arviz-devs/arviz/pull/1588))
-   Correctly use chain index when constructing PyMC3 `DefaultTrace` in `from_pymc3` ([1590](https://github.com/arviz-devs/arviz/pull/1590))
-   Fix bugs in CmdStanPyConverter ([1595](https://github.com/arviz-devs/arviz/pull/1595) and [1598](https://github.com/arviz-devs/arviz/pull/1598))
-   Fix `c` argument in `plot_khat` ([1592](https://github.com/arviz-devs/arviz/pull/1592))
-   Fix `ax` argument in `plot_elpd` ([1593](https://github.com/arviz-devs/arviz/pull/1593))
-   Remove warning in `stats.py` compare function ([1607](https://github.com/arviz-devs/arviz/pull/1607))
-   Fix `ess/rhat` plots in `plot_forest` ([1606](https://github.com/arviz-devs/arviz/pull/1606))
-   Fix `from_numpyro` crash when importing model with `thinning=x` for `x > 1` ([1619](https://github.com/arviz-devs/arviz/pull/1619))
-   Upload updated mypy.ini in ci if mypy copilot fails ([1624](https://github.com/arviz-devs/arviz/pull/1624))
-   Added type checking to raise an error whenever `InferenceData` object is passed using `io_pymc3`'s `trace` argument ([1629](https://github.com/arviz-devs/arviz/pull/1629))
-   Fix `xlabels` in `plot_elpd` ([1601](https://github.com/arviz-devs/arviz/pull/1601))
-   Renamed `sample` dim to `__sample__` when stacking `chain` and `draw` to avoid dimension collision ([1647](https://github.com/arviz-devs/arviz/pull/1647))
-   Removed the `circular` argument in `plot_dist` in favor of `is_circular` ([1681](https://github.com/arviz-devs/arviz/pull/1681))
-   Fix `legend` argument in `plot_separation` ([1701](https://github.com/arviz-devs/arviz/pull/1701))
-   Removed testing dependency on http download for radon dataset ([1717](https://github.com/arviz-devs/arviz/pull/1717))
-   Fixed plot_kde to take labels with kwargs. ([1710](https://github.com/arviz-devs/arviz/pull/1710))
-   Fixed xarray related tests. ([1726](https://github.com/arviz-devs/arviz/pull/1726))
-   Fix Bokeh deprecation warnings ([1657](https://github.com/arviz-devs/arviz/pull/1657))
-   Fix credible inteval percentage in legend in `plot_loo_pit` ([1745](https://github.com/arviz-devs/arviz/pull/1745))
-   Arguments `filter_vars` and `filter_groups` now raise `ValueError` if illegal arguments are passed ([1772](https://github.com/arviz-devs/arviz/pull/1772))
-   Remove constrained_layout from arviz rcparams ([1764](https://github.com/arviz-devs/arviz/pull/1764))
-   Fix plot_elpd for a single outlier ([1787](https://github.com/arviz-devs/arviz/pull/1787))

### Deprecation

-   Deprecated `index_origin` and `order` arguments in `az.summary` ([1201](https://github.com/arviz-devs/arviz/pull/1201))

### Documentation

-   Language improvements of the first third of the "Label guide" ([1699](https://github.com/arviz-devs/arviz/pull/1699))
-   Added "Label guide" page and API section for `arviz.labels` module ([1201](https://github.com/arviz-devs/arviz/pull/1201) and [1635](https://github.com/arviz-devs/arviz/pull/1635))
-   Add "Installation guide" page to the documentation ([1551](https://github.com/arviz-devs/arviz/pull/1551))
-   Improve documentation on experimental `SamplingWrapper` classes ([1582](https://github.com/arviz-devs/arviz/pull/1582))
-   Added example to `plot_hdi` using Inference Data ([1615](https://github.com/arviz-devs/arviz/pull/1615))
-   Removed `geweke` diagnostic from `numba` user guide ([1653](https://github.com/arviz-devs/arviz/pull/1653))
-   Restructured the documentation sections to improve community and about us information ([1587](https://github.com/arviz-devs/arviz/pull/1587))

## v0.11.2 (2021 Feb 21)

### New features

-   Added `to_zarr` and `from_zarr` methods to InferenceData ([1518](https://github.com/arviz-devs/arviz/pull/1518))
-   Added confidence interval band to auto-correlation plot ([1535](https://github.com/arviz-devs/arviz/pull/1535))

### Maintenance and fixes

-   Updated CmdStanPy converter form compatibility with versions >=0.9.68 ([1558](https://github.com/arviz-devs/arviz/pull/1558) and ([1564](https://github.com/arviz-devs/arviz/pull/1564))
-   Updated `from_cmdstanpy`, `from_cmdstan`, `from_numpyro` and `from_pymc3` converters to follow schema convention ([1550](https://github.com/arviz-devs/arviz/pull/1550), [1541](https://github.com/arviz-devs/arviz/pull/1541), [1525](https://github.com/arviz-devs/arviz/pull/1525) and [1555](https://github.com/arviz-devs/arviz/pull/1555))
-   Fix calculation of mode as point estimate ([1552](https://github.com/arviz-devs/arviz/pull/1552))
-   Remove variable name from legend in posterior predictive plot ([1559](https://github.com/arviz-devs/arviz/pull/1559))
-   Added significant digits formatter to round rope values ([1569](https://github.com/arviz-devs/arviz/pull/1569))
-   Updated `from_cmdstan`. csv reader, dtype problem fixed and dtype kwarg added for manual dtype casting ([1565](https://github.com/arviz-devs/arviz/pull/1565))

### Deprecation

-   Removed Geweke diagnostic ([1545](https://github.com/arviz-devs/arviz/pull/1545))
-   Removed credible_interval and include_circ arguments ([1548](https://github.com/arviz-devs/arviz/pull/1548))

### Documentation

-   Added an example for converting dataframe to InferenceData ([1556](https://github.com/arviz-devs/arviz/pull/1556))
-   Added example for `coords` argument in `plot_posterior` docstring ([1566](https://github.com/arviz-devs/arviz/pull/1566))

## v0.11.1 (2021 Feb 2)

### Maintenance and fixes

-   Fixed ovelapping titles and repeating warnings on circular traceplot ([1517](https://github.com/arviz-devs/arviz/pull/1517))
-   Removed repetitive variable names from forest plots of multivariate variables ([1527](https://github.com/arviz-devs/arviz/pull/1527))
-   Fixed regression in `plot_pair` labels that prevented coord names to be shown when necessary ([1533](https://github.com/arviz-devs/arviz/pull/1533))

### Documentation

-   Use tabs in ArviZ example gallery ([1521](https://github.com/arviz-devs/arviz/pull/1521))

## v0.11.0 (2021 Dec 17)

### New features

-   Added `to_dataframe` method to InferenceData ([1395](https://github.com/arviz-devs/arviz/pull/1395))
-   Added `__getitem__` magic to InferenceData ([1395](https://github.com/arviz-devs/arviz/pull/1395))
-   Added group argument to summary ([1408](https://github.com/arviz-devs/arviz/pull/1408))
-   Add `ref_line`, `bar`, `vlines` and `marker_vlines` kwargs to `plot_rank` ([1419](https://github.com/arviz-devs/arviz/pull/1419))
-   Add observed argument to (un)plot observed data in `plot_ppc` ([1422](https://github.com/arviz-devs/arviz/pull/1422))
-   Add support for named dims and coordinates with multivariate observations ([1429](https://github.com/arviz-devs/arviz/pull/1429))
-   Add support for discrete variables in rank plots ([1433](https://github.com/arviz-devs/arviz/pull/1433)) and
    `loo_pit` ([1500](https://github.com/arviz-devs/arviz/pull/1500))
-   Add `skipna` argument to `plot_posterior` ([1432](https://github.com/arviz-devs/arviz/pull/1432))
-   Make stacking the default method to compute weights in `compare` ([1438](https://github.com/arviz-devs/arviz/pull/1438))
-   Add `copy()` method to `InferenceData` class. ([1501](https://github.com/arviz-devs/arviz/pull/1501)).

### Maintenance and fixes

-   prevent wrapping group names in InferenceData repr_html ([1407](https://github.com/arviz-devs/arviz/pull/1407))
-   Updated CmdStanPy interface ([1409](https://github.com/arviz-devs/arviz/pull/1409))
-   Remove left out warning about default IC scale in `compare` ([1412](https://github.com/arviz-devs/arviz/pull/1412))
-   Fixed a typo found in an error message raised in `distplot.py` ([1414](https://github.com/arviz-devs/arviz/pull/1414))
-   Fix typo in `loo_pit` extraction of log likelihood ([1418](https://github.com/arviz-devs/arviz/pull/1418))
-   Have `from_pystan` store attrs as strings to allow netCDF storage ([1417](https://github.com/arviz-devs/arviz/pull/1417))
-   Remove ticks and spines in `plot_violin` ([1426 ](https://github.com/arviz-devs/arviz/pull/1426))
-   Use circular KDE function and fix tick labels in circular `plot_trace` ([1428](https://github.com/arviz-devs/arviz/pull/1428))
-   Fix `pair_plot` for mixed discrete and continuous variables ([1434](https://github.com/arviz-devs/arviz/pull/1434))
-   Fix in-sample deviance in `plot_compare` ([1435](https://github.com/arviz-devs/arviz/pull/1435))
-   Fix computation of weights in compare ([1438](https://github.com/arviz-devs/arviz/pull/1438))
-   Avoid repeated warning in summary ([1442](https://github.com/arviz-devs/arviz/pull/1442))
-   Fix hdi failure with boolean array ([1444](https://github.com/arviz-devs/arviz/pull/1444))
-   Automatically get the current axes instance for `plt_kde`, `plot_dist` and `plot_hdi` ([1452](https://github.com/arviz-devs/arviz/pull/1452))
-   Add grid argument to manually specify the number of rows and columns ([1459](https://github.com/arviz-devs/arviz/pull/1459))
-   Switch to `compact=True` by default in our plots ([1468](https://github.com/arviz-devs/arviz/issues/1468))
-   `plot_elpd`, avoid modifying the input dict ([1477](https://github.com/arviz-devs/arviz/issues/1477))
-   Do not plot divergences in `plot_trace` when `kind=rank_vlines` or `kind=rank_bars` ([1476](https://github.com/arviz-devs/arviz/issues/1476))
-   Allow ignoring `observed` argument of `pymc3.DensityDist` in `from_pymc3` ([1495](https://github.com/arviz-devs/arviz/pull/1495))
-   Make `from_pymc3` compatible with theano-pymc 1.1.0 ([1495](https://github.com/arviz-devs/arviz/pull/1495))
-   Improve typing hints ([1491](https://github.com/arviz-devs/arviz/pull/1491), ([1492](https://github.com/arviz-devs/arviz/pull/1492),
    ([1493](https://github.com/arviz-devs/arviz/pull/1493), ([1494](https://github.com/arviz-devs/arviz/pull/1494) and
    ([1497](https://github.com/arviz-devs/arviz/pull/1497))

### Deprecation

-   `plot_khat` deprecate `annotate` argument in favor of `threshold`. The new argument accepts floats ([1478](https://github.com/arviz-devs/arviz/issues/1478))

### Documentation

-   Reorganize documentation and change sphinx theme ([1406](https://github.com/arviz-devs/arviz/pull/1406))
-   Switch to [MyST](https://myst-parser.readthedocs.io/en/latest/) and [MyST-NB](https://myst-nb.readthedocs.io/en/latest/index.html)
    for markdown/notebook parsing in docs ([1406](https://github.com/arviz-devs/arviz/pull/1406))
-   Incorporated `input_core_dims` in `hdi` and `plot_hdi` docstrings ([1410](https://github.com/arviz-devs/arviz/pull/1410))
-   Add documentation pages about experimental `SamplingWrapper`s usage ([1373](https://github.com/arviz-devs/arviz/pull/1373))
-   Show example titles in gallery page ([1484](https://github.com/arviz-devs/arviz/pull/1484))
-   Add `sample_stats` naming convention to the InferenceData schema ([1063](https://github.com/arviz-devs/arviz/pull/1063))
-   Extend api documentation about `InferenceData` methods ([1338](https://github.com/arviz-devs/arviz/pull/1338))

### Experimental

-   Modified `SamplingWrapper` base API ([1373](https://github.com/arviz-devs/arviz/pull/1373))

## v0.10.0 (2020 Sep 24)

### New features

-   Added InferenceData dataset containing circular variables ([1265](https://github.com/arviz-devs/arviz/pull/1265))
-   Added `is_circular` argument to `plot_dist` and `plot_kde` allowing for a circular histogram (Matplotlib, Bokeh) or 1D KDE plot (Matplotlib). ([1266](https://github.com/arviz-devs/arviz/pull/1266))
-   Added `to_dict` method for InferenceData object ([1223](https://github.com/arviz-devs/arviz/pull/1223))
-   Added `circ_var_names` argument to `plot_trace` allowing for circular traceplot (Matplotlib) ([1336](https://github.com/arviz-devs/arviz/pull/1336))
-   Ridgeplot is hdi aware. By default displays truncated densities at the specified `hdi_prop` level ([1348](https://github.com/arviz-devs/arviz/pull/1348))
-   Added `plot_separation` ([1359](https://github.com/arviz-devs/arviz/pull/1359))
-   Extended methods from `xr.Dataset` to `InferenceData` ([1254](https://github.com/arviz-devs/arviz/pull/1254))
-   Add `extend` and `add_groups` to `InferenceData` ([1300](https://github.com/arviz-devs/arviz/pull/1300) and [1386](https://github.com/arviz-devs/arviz/pull/1386))
-   Added `__iter__` method (`.items`) for InferenceData ([1356](https://github.com/arviz-devs/arviz/pull/1356))
-   Add support for discrete variables in `plot_bpv` ([#1379](https://github.com/arviz-devs/arviz/pull/1379))

### Maintenance and fixes

-   Automatic conversion of list/tuple to numpy array in distplot ([1277](https://github.com/arviz-devs/arviz/pull/1277))
-   `plot_posterior` fix overlap of hdi and rope ([1263](https://github.com/arviz-devs/arviz/pull/1263))
-   `plot_dist` bins argument error fixed ([1306](https://github.com/arviz-devs/arviz/pull/1306))
-   Improve handling of circular variables in `az.summary` ([1313](https://github.com/arviz-devs/arviz/pull/1313))
-   Removed change of default warning in `ELPDData` string representation ([1321](https://github.com/arviz-devs/arviz/pull/1321))
-   Update `radon` example dataset to current InferenceData schema specification ([1320](https://github.com/arviz-devs/arviz/pull/1320))
-   Update `from_cmdstan` functionality and add warmup groups ([1330](https://github.com/arviz-devs/arviz/pull/1330) and [1351](https://github.com/arviz-devs/arviz/pull/1351))
-   Restructure plotting code to be compatible with mpl>=3.3 ([1312](https://github.com/arviz-devs/arviz/pull/1312) and [1352](https://github.com/arviz-devs/arviz/pull/1352))
-   Replaced `_fast_kde()` with `kde()` which now also supports circular variables via the argument `circular` ([1284](https://github.com/arviz-devs/arviz/pull/1284)).
-   Increased `from_pystan` attrs information content ([1353](https://github.com/arviz-devs/arviz/pull/1353))
-   Allow `plot_trace` to return and accept axes ([1361](https://github.com/arviz-devs/arviz/pull/1361))
-   Update diagnostics to be on par with posterior package ([1366](https://github.com/arviz-devs/arviz/pull/1366))
-   Use method="average" in `scipy.stats.rankdata` ([1380](https://github.com/arviz-devs/arviz/pull/1380))
-   Add more `plot_parallel` examples ([1380](https://github.com/arviz-devs/arviz/pull/1380))
-   Bump minimum xarray version to 0.16.1 ([1389](https://github.com/arviz-devs/arviz/pull/1389)
-   Fix multi rope for `plot_forest` ([1390](https://github.com/arviz-devs/arviz/pull/1390))
-   Bump minimum xarray version to 0.16.1 ([1389](https://github.com/arviz-devs/arviz/pull/1389))
-   `from_dict` will now store warmup groups even with the main group missing ([1386](https://github.com/arviz-devs/arviz/pull/1386))
-   increase robustness for repr_html handling ([1392](https://github.com/arviz-devs/arviz/pull/1392))

## v0.9.0 (2020 June 23)

### New features

-   loo-pit plot. The kde is computed over the data interval (this could be shorter than [0, 1]). The HDI is computed analytically ([1215](https://github.com/arviz-devs/arviz/pull/1215))
-   Added `html_repr` of InferenceData objects for jupyter notebooks. ([1217](https://github.com/arviz-devs/arviz/pull/1217))
-   Added support for PyJAGS via the function `from_pyjags`. ([1219](https://github.com/arviz-devs/arviz/pull/1219) and [1245](https://github.com/arviz-devs/arviz/pull/1245))
-   `from_pymc3` can now retrieve `coords` and `dims` from model context ([1228](https://github.com/arviz-devs/arviz/pull/1228), [1240](https://github.com/arviz-devs/arviz/pull/1240) and [1249](https://github.com/arviz-devs/arviz/pull/1249))
-   `plot_trace` now supports multiple aesthetics to identify chain and variable
    shape and support matplotlib aliases ([1253](https://github.com/arviz-devs/arviz/pull/1253))
-   `plot_hdi` can now take already computed HDI values ([1241](https://github.com/arviz-devs/arviz/pull/1241))
-   `plot_bpv`. A new plot for Bayesian p-values ([1222](https://github.com/arviz-devs/arviz/pull/1222))

### Maintenance and fixes

-   Include data from `MultiObservedRV` to `observed_data` when using
    `from_pymc3` ([1098](https://github.com/arviz-devs/arviz/pull/1098))
-   Added a note on `plot_pair` when trying to use `plot_kde` on `InferenceData`
    objects. ([1218](https://github.com/arviz-devs/arviz/pull/1218))
-   Added `log_likelihood` argument to `from_pyro` and a warning if log likelihood cannot be obtained ([1227](https://github.com/arviz-devs/arviz/pull/1227))
-   Skip tests on matplotlib animations if ffmpeg is not installed ([1227](https://github.com/arviz-devs/arviz/pull/1227))
-   Fix hpd bug where arguments were being ignored ([1236](https://github.com/arviz-devs/arviz/pull/1236))
-   Remove false positive warning in `plot_hdi` and fixed matplotlib axes generation ([1241](https://github.com/arviz-devs/arviz/pull/1241))
-   Change the default `zorder` of scatter points from `0` to `0.6` in `plot_pair` ([1246](https://github.com/arviz-devs/arviz/pull/1246))
-   Update `get_bins` for numpy 1.19 compatibility ([1256](https://github.com/arviz-devs/arviz/pull/1256))
-   Fixes to `rug`, `divergences` arguments in `plot_trace` ([1253](https://github.com/arviz-devs/arviz/pull/1253))

### Deprecation

-   Using `from_pymc3` without a model context available now raises a
    `FutureWarning` and will be deprecated in a future version ([1227](https://github.com/arviz-devs/arviz/pull/1227))
-   In `plot_trace`, `chain_prop` and `compact_prop` as tuples will now raise a
    `FutureWarning` ([1253](https://github.com/arviz-devs/arviz/pull/1253))
-   `hdi` with 2d data raises a FutureWarning ([1241](https://github.com/arviz-devs/arviz/pull/1241))

### Documentation

-   A section has been added to the documentation at InferenceDataCookbook.ipynb illustrating the use of ArviZ in conjunction with PyJAGS. ([1219](https://github.com/arviz-devs/arviz/pull/1219) and [1245](https://github.com/arviz-devs/arviz/pull/1245))
-   Fixed inconsistent capitalization in `plot_hdi` docstring ([1221](https://github.com/arviz-devs/arviz/pull/1221))
-   Fixed and extended `InferenceData.map` docs ([1255](https://github.com/arviz-devs/arviz/pull/1255))

## v0.8.3 (2020 May 28)

### Maintenance and fixes

-   Restructured internals of `from_pymc3` to handle old pymc3 releases and
    sliced traces and to provide useful warnings ([1211](https://github.com/arviz-devs/arviz/pull/1211))

## v0.8.2 (2020 May 25)

### Maintenance and fixes

-   Fixed bug in `from_pymc3` for sliced `pymc3.MultiTrace` input ([1209](https://github.com/arviz-devs/arviz/pull/1209))

## v0.8.1 (2020 May 24)

### Maintenance and fixes

-   Fixed bug in `from_pymc3` when used with PyMC3<3.9 ([1203](https://github.com/arviz-devs/arviz/pull/1203))
-   Fixed enforcement of rcParam `plot.max_subplots` in `plot_trace` and
    `plot_pair` ([1205](https://github.com/arviz-devs/arviz/pull/1205))
-   Removed extra subplot row and column in in `plot_pair` with `marginal=True` ([1205](https://github.com/arviz-devs/arviz/pull/1205))
-   Added latest PyMC3 release to CI in addition to using GitHub default branch ([1207](https://github.com/arviz-devs/arviz/pull/1207))

### Documentation

-   Use `dev` as version indicator in online documentation ([1204](https://github.com/arviz-devs/arviz/pull/1204))

## v0.8.0 (2020 May 23)

### New features

-   Stats and plotting functions that provide `var_names` arg can now filter parameters based on partial naming (`filter="like"`) or regular expressions (`filter="regex"`) (see [1154](https://github.com/arviz-devs/arviz/pull/1154)).
-   Add `true_values` argument for `plot_pair`. It allows for a scatter plot showing the true values of the variables ([1140](https://github.com/arviz-devs/arviz/pull/1140))
-   Allow xarray.Dataarray input for plots.([1120](https://github.com/arviz-devs/arviz/pull/1120))
-   Revamped the `hpd` function to make it work with mutidimensional arrays, InferenceData and xarray objects ([1117](https://github.com/arviz-devs/arviz/pull/1117))
-   Skip test for optional/extra dependencies when not installed ([1113](https://github.com/arviz-devs/arviz/pull/1113))
-   Add option to display rank plots instead of trace ([1134](https://github.com/arviz-devs/arviz/pull/1134))
-   Add out-of-sample groups (`predictions` and `predictions_constant_data`) to `from_dict` ([1125](https://github.com/arviz-devs/arviz/pull/1125))
-   Add out-of-sample groups (`predictions` and `predictions_constant_data`) and `constant_data` group to pyro and numpyro translation ([1090](https://github.com/arviz-devs/arviz/pull/1090), [1125](https://github.com/arviz-devs/arviz/pull/1125))
-   Add `num_chains` and `pred_dims` arguments to from_pyro and from_numpyro ([1090](https://github.com/arviz-devs/arviz/pull/1090), [1125](https://github.com/arviz-devs/arviz/pull/1125))
-   Integrate jointplot into pairplot, add point-estimate and overlay of plot kinds ([1079](https://github.com/arviz-devs/arviz/pull/1079))
-   New grayscale style. This also add two new cmaps `cet_grey_r` and `cet_grey_r`. These are perceptually uniform gray scale cmaps from colorcet (linear_grey_10_95_c0) ([1164](https://github.com/arviz-devs/arviz/pull/1164))
-   Add warmup groups to InferenceData objects, initial support for PyStan ([1126](https://github.com/arviz-devs/arviz/pull/1126)) and PyMC3 ([1171](https://github.com/arviz-devs/arviz/pull/1171))
-   `hdi_prob` will not plot hdi if argument `hide` is passed. Previously `credible_interval` would omit HPD if `None` was passed ([1176](https://github.com/arviz-devs/arviz/pull/1176))
-   Add `stats.ic_pointwise` rcParam ([1173](https://github.com/arviz-devs/arviz/pull/1173))
-   Add `var_name` argument to information criterion calculation: `compare`,
    `loo` and `waic` ([1173](https://github.com/arviz-devs/arviz/pull/1173))

### Maintenance and fixes

-   Fixed `plot_pair` functionality for two variables with bokeh backend ([1179](https://github.com/arviz-devs/arviz/pull/1179))
-   Changed `diagonal` argument for `marginals` and fixed `point_estimate_marker_kwargs` in `plot_pair` ([1167](https://github.com/arviz-devs/arviz/pull/1167))
-   Fixed behaviour of `credible_interval=None` in `plot_posterior` ([1115](https://github.com/arviz-devs/arviz/pull/1115))
-   Fixed hist kind of `plot_dist` with multidimensional input ([1115](https://github.com/arviz-devs/arviz/pull/1115))
-   Fixed `TypeError` in `transform` argument of `plot_density` and `plot_forest` when `InferenceData` is a list or tuple ([1121](https://github.com/arviz-devs/arviz/pull/1121))
-   Fixed overlaid pairplots issue ([1135](https://github.com/arviz-devs/arviz/pull/1135))
-   Update Docker building steps ([1127](https://github.com/arviz-devs/arviz/pull/1127))
-   Updated benchmarks and moved to asv_benchmarks/benchmarks ([1142](https://github.com/arviz-devs/arviz/pull/1142))
-   Moved `_fast_kde`, `_fast_kde_2d`, `get_bins` and `_sturges_formula` to `numeric_utils` and `get_coords` to `utils` ([1142](https://github.com/arviz-devs/arviz/pull/1142))
-   Rank plot: rename `axes` argument to `ax` ([1144](https://github.com/arviz-devs/arviz/pull/1144))
-   Added a warning specifying log scale is now the default in compare/loo/waic functions ([1150](https://github.com/arviz-devs/arviz/pull/1150)).
-   Fixed bug in `plot_posterior` with rcParam "plot.matplotlib.show" = True ([1151](https://github.com/arviz-devs/arviz/pull/1151))
-   Set `fill_last` argument of `plot_kde` to False by default ([1158](https://github.com/arviz-devs/arviz/pull/1158))
-   plot_ppc animation: improve docs and error handling ([1162](https://github.com/arviz-devs/arviz/pull/1162))
-   Fix import error when wrapped function docstring is empty ([1192](https://github.com/arviz-devs/arviz/pull/1192))
-   Fix passing axes to plot_density with several datasets ([1198](https://github.com/arviz-devs/arviz/pull/1198))

### Deprecation

-   `hpd` function deprecated in favor of `hdi`. `credible_interval` argument replaced by `hdi_prob`throughout with exception of `plot_loo_pit` ([1176](https://github.com/arviz-devs/arviz/pull/1176))
-   `plot_hpd` function deprecated in favor of `plot_hdi`. ([1190](https://github.com/arviz-devs/arviz/pull/1190))

### Documentation

-   Add classifier to `setup.py` including Matplotlib framework ([1133](https://github.com/arviz-devs/arviz/pull/1133))
-   Image thumbs generation updated to be Bokeh 2 compatible ([1116](https://github.com/arviz-devs/arviz/pull/1116))
-   Add new examples for `plot_pair` ([1110](https://github.com/arviz-devs/arviz/pull/1110))
-   Add examples for `psislw` and `r2_score` ([1129](https://github.com/arviz-devs/arviz/pull/1129))
-   Add more examples on 2D kde customization ([1158](https://github.com/arviz-devs/arviz/pull/1158))
-   Make docs compatible with sphinx3 and configure `intersphinx` for better
    references ([1184](https://github.com/arviz-devs/arviz/pull/1184))
-   Extend the developer guide and add it to the website ([1184](https://github.com/arviz-devs/arviz/pull/1184))

## v0.7.0 (2020 Mar 2)

### New features

-   Add out-of-sample predictions (`predictions` and `predictions_constant_data` groups) to pymc3, pystan, cmdstan and cmdstanpy translations ([983](https://github.com/arviz-devs/arviz/pull/983), [1032](https://github.com/arviz-devs/arviz/pull/1032) and [1064](https://github.com/arviz-devs/arviz/pull/1064))
-   Started adding pointwise log likelihood storage support ([794](https://github.com/arviz-devs/arviz/pull/794), [1044](https://github.com/arviz-devs/arviz/pull/1044) and [1064](https://github.com/arviz-devs/arviz/pull/1064))
-   Add out-of-sample predictions (`predictions` and `predictions_constant_data` groups) to pymc3 and pystan translations ([983](https://github.com/arviz-devs/arviz/pull/983) and [1032](https://github.com/arviz-devs/arviz/pull/1032))
-   Started adding pointwise log likelihood storage support ([794](https://github.com/arviz-devs/arviz/pull/794), [1044](https://github.com/arviz-devs/arviz/pull/1044))
-   Violinplot: rug-plot option ([997](https://github.com/arviz-devs/arviz/pull/997))
-   Integrated rcParams `plot.point_estimate` ([994](https://github.com/arviz-devs/arviz/pull/994)), `stats.ic_scale` ([993](https://github.com/arviz-devs/arviz/pull/993)) and `stats.credible_interval` ([1017](https://github.com/arviz-devs/arviz/pull/1017))
-   Added `group` argument to `plot_ppc` ([1008](https://github.com/arviz-devs/arviz/pull/1008)), `plot_pair` ([1009](https://github.com/arviz-devs/arviz/pull/1009)) and `plot_joint` ([1012](https://github.com/arviz-devs/arviz/pull/1012))
-   Added `transform` argument to `plot_trace`, `plot_forest`, `plot_pair`, `plot_posterior`, `plot_rank`, `plot_parallel`, `plot_violin`,`plot_density`, `plot_joint` ([1036](https://github.com/arviz-devs/arviz/pull/1036))
-   Add `skipna` argument to `hpd` and `summary` ([1035](https://github.com/arviz-devs/arviz/pull/1035))
-   Added `transform` argument to `plot_trace`, `plot_forest`, `plot_pair`, `plot_posterior`, `plot_rank`, `plot_parallel`, `plot_violin`,`plot_density`, `plot_joint` ([1036](https://github.com/arviz-devs/arviz/pull/1036))
-   Add `marker` functionality to `bokeh_plot_elpd` ([1040](https://github.com/arviz-devs/arviz/pull/1040))
-   Add `ridgeplot_quantiles` argument to `plot_forest` ([1047](https://github.com/arviz-devs/arviz/pull/1047))
-   Added the functionality [interactive legends](https://docs.bokeh.org/en/1.4.0/docs/user_guide/interaction/legends.html) for bokeh plots of `densityplot`, `energyplot`
    and `essplot` ([1024](https://github.com/arviz-devs/arviz/pull/1024))
-   New defaults for cross validation: `loo` (old: waic) and `log` -scale (old: `deviance` -scale) ([1067](https://github.com/arviz-devs/arviz/pull/1067))
-   **Experimental Feature**: Added `arviz.wrappers` module to allow ArviZ to refit the models if necessary ([771](https://github.com/arviz-devs/arviz/pull/771))
-   **Experimental Feature**: Added `reloo` function to ArviZ ([771](https://github.com/arviz-devs/arviz/pull/771))
-   Added new helper function `matplotlib_kwarg_dealiaser` ([1073](https://github.com/arviz-devs/arviz/pull/1073))
-   ArviZ version to InferenceData attributes. ([1086](https://github.com/arviz-devs/arviz/pull/1086))
-   Add `log_likelihood` argument to `from_pymc3` ([1082](https://github.com/arviz-devs/arviz/pull/1082))
-   Integrated rcParams for `plot.bokeh.layout` and `plot.backend`. ([1089](https://github.com/arviz-devs/arviz/pull/1089))
-   Add automatic legends in `plot_trace` with compact=True (matplotlib only) ([1070](https://github.com/arviz-devs/arviz/pull/1070))
-   Updated hover information for `plot_pair` with bokeh backend ([1074](https://github.com/arviz-devs/arviz/pull/1074))

### Maintenance and fixes

-   Fixed bug in density and posterior plot bin computation ([1049](https://github.com/arviz-devs/arviz/pull/1049))
-   Fixed bug in density plot ax argument ([1049](https://github.com/arviz-devs/arviz/pull/1049))
-   Fixed bug in extracting prior samples for cmdstanpy ([979](https://github.com/arviz-devs/arviz/pull/979))
-   Fix erroneous warning in traceplot ([989](https://github.com/arviz-devs/arviz/pull/989))
-   Correct bfmi denominator ([991](https://github.com/arviz-devs/arviz/pull/991))
-   Removed parallel from jit full ([996](https://github.com/arviz-devs/arviz/pull/996))
-   Rename flat_inference_data_to_dict ([1003](https://github.com/arviz-devs/arviz/pull/1003))
-   Violinplot: fix histogram ([997](https://github.com/arviz-devs/arviz/pull/997))
-   Convert all instances of SyntaxWarning to UserWarning ([1016](https://github.com/arviz-devs/arviz/pull/1016))
-   Fix `point_estimate` in `plot_posterior` ([1038](https://github.com/arviz-devs/arviz/pull/1038))
-   Fix interpolation `hpd_plot` ([1039](https://github.com/arviz-devs/arviz/pull/1039))
-   Fix `io_pymc3.py` to handle models with `potentials` ([1043](https://github.com/arviz-devs/arviz/pull/1043))
-   Fix several inconsistencies between schema and `from_pymc3` implementation
    in groups `prior`, `prior_predictive` and `observed_data` ([1045](https://github.com/arviz-devs/arviz/pull/1045))
-   Stabilize covariance matrix for `plot_kde_2d` ([1075](https://github.com/arviz-devs/arviz/pull/1075))
-   Removed extra dim in `prior` data in `from_pyro` ([1071](https://github.com/arviz-devs/arviz/pull/1071))
-   Moved CI and docs (build & deploy) to Azure Pipelines and started using codecov ([1080](https://github.com/arviz-devs/arviz/pull/1080))
-   Fixed bug in densityplot when variables differ between models ([1096](https://github.com/arviz-devs/arviz/pull/1096))

### Deprecation

-   `from_pymc3` now requires PyMC3>=3.8

### Documentation

-   Updated `InferenceData` schema specification (`log_likelihood`,
    `predictions` and `predictions_constant_data` groups)
-   Clarify the usage of `plot_joint` ([1001](https://github.com/arviz-devs/arviz/pull/1001))
-   Added the API link of function to examples ([1013](https://github.com/arviz-devs/arviz/pull/1013))
-   Updated PyStan_schema_example to include example of out-of-sample prediction ([1032](https://github.com/arviz-devs/arviz/pull/1032))
-   Added example for `concat` method ([1037](https://github.com/arviz-devs/arviz/pull/1037))

## v0.6.1 (2019 Dec 28)

### New features

-   Update for pair_plot divergences can be selected
-   Default tools follow global (ArviZ) defaults
-   Add interactive legend for a plot, if only two variables are used in pairplot

### Maintenance and fixes

-   Change `packaging` import from absolute to relative format, explicitly importing `version` function

## v0.6.0 (2019 Dec 24)

### New features

-   Initial bokeh support.
-   ArviZ.jl a Julia interface to ArviZ (@sethaxen )

### Maintenance and fixes

-   Fully support `numpyro` (@fehiepsi )
-   log_likelihood and observed data from pyro
-   improve rcparams
-   fix `az.concat` functionality (@anzelpwj )

### Documentation

-   distplot docstring plotting example (@jscarbor )

## v0.5.1 (2019 Sep 16)

### Maintenance and fixes

-   Comment dev requirements in setup.py

## v0.5.0 (2019 Sep 15)

## New features

-   Add from_numpyro Integration ([811](https://github.com/arviz-devs/arviz/pull/811))
-   Numba Google Summer of Code additions (https://ban-zee.github.io/jekyll/update/2019/08/19/Submission.html)
-   Model checking, Inference Data, and Convergence assessments (https://github.com/OriolAbril/gsoc2019/blob/master/final_work_submission.md)

## v0.4.1 (2019 Jun 9)

### New features

-   Reorder stats columns ([695](https://github.com/arviz-devs/arviz/pull/695))
-   Plot Forest reports ess and rhat by default([685](https://github.com/arviz-devs/arviz/pull/685))
-   Add pointwise elpd ([678](https://github.com/arviz-devs/arviz/pull/678))

### Maintenance and fixes

-   Fix io_pymc3 bug ([693](https://github.com/arviz-devs/arviz/pull/693))
-   Fix io_pymc3 warning ([686](https://github.com/arviz-devs/arviz/pull/686))
-   Fix 0 size bug with pystan ([677](https://github.com/arviz-devs/arviz/pull/677))

## v0.4.0 (2019 May 20)

### New features

-   Add plot_dist ([592](https://github.com/arviz-devs/arviz/pull/592))
-   New rhat and ess ([623](https://github.com/arviz-devs/arviz/pull/623))
-   Add plot_hpd ([611](https://github.com/arviz-devs/arviz/pull/611))
-   Add plot_rank ([625](https://github.com/arviz-devs/arviz/pull/625))

### Deprecation

-   Remove load_data and save_data ([625](https://github.com/arviz-devs/arviz/pull/625))

## v0.3.3 (2019 Feb 23)

### New features

-   Plot ppc supports multiple chains ([526](https://github.com/arviz-devs/arviz/pull/526))
-   Plot titles now wrap ([441](https://github.com/arviz-devs/arviz/pull/441))
-   plot_density uses a grid ([379](https://github.com/arviz-devs/arviz/pull/379))
-   emcee reader support ([550](https://github.com/arviz-devs/arviz/pull/550))
-   Animations in plot_ppc ([546](https://github.com/arviz-devs/arviz/pull/546))
-   Optional dictionary for stat_funcs in summary ([583](https://github.com/arviz-devs/arviz/pull/583))
-   Can exclude variables in selections with negated variable names ([574](https://github.com/arviz-devs/arviz/pull/574))

### Maintenance and fixes

-   Order maintained with xarray_var_iter ([557](https://github.com/arviz-devs/arviz/pull/557))
-   Testing very improved (multiple)
-   Fix nan handling in effective sample size ([573](https://github.com/arviz-devs/arviz/pull/573))
-   Fix kde scaling ([582](https://github.com/arviz-devs/arviz/pull/582))
-   xticks for discrete variables ([586](https://github.com/arviz-devs/arviz/pull/586))
-   Empty InferenceData saves consistent with netcdf ([577](https://github.com/arviz-devs/arviz/pull/577))
-   Removes numpy pinning ([594](https://github.com/arviz-devs/arviz/pull/594))

### Documentation

-   JOSS and Zenodo badges ([537](https://github.com/arviz-devs/arviz/pull/537))
-   Gitter badge ([548](https://github.com/arviz-devs/arviz/pull/548))
-   Docs for combining InferenceData ([590](https://github.com/arviz-devs/arviz/pull/590))

## v0.3.2 (2019 Jan 15)

### New features

-   Support PyStan3 ([464](https://github.com/arviz-devs/arviz/pull/464))
-   Add some more information to the inference data of tfp ([447](https://github.com/arviz-devs/arviz/pull/447))
-   Use Split R-hat ([477](https://github.com/arviz-devs/arviz/pull/477))
-   Normalize from_xyz functions ([490](https://github.com/arviz-devs/arviz/pull/490))
-   KDE: Display quantiles ([479](https://github.com/arviz-devs/arviz/pull/479))
-   Add multiple rope support to `plot_forest` ([448](https://github.com/arviz-devs/arviz/pull/448))
-   Numba jit compilation to speed up some methods ([515](https://github.com/arviz-devs/arviz/pull/515))
-   Add `from_dict` for easier creation of az.InferenceData objects ([524](https://github.com/arviz-devs/arviz/pull/524))
-   Add stable logsumexp ([522](https://github.com/arviz-devs/arviz/pull/522))

### Maintenance and fixes

-   Fix for `from_pyro` with multiple chains ([463](https://github.com/arviz-devs/arviz/pull/463))
-   Check `__version__` for attr ([466](https://github.com/arviz-devs/arviz/pull/466))
-   And exception to plot compare ([461](https://github.com/arviz-devs/arviz/pull/461))
-   Add Docker Testing to travisCI ([473](https://github.com/arviz-devs/arviz/pull/473))
-   fix jointplot warning ([478](https://github.com/arviz-devs/arviz/pull/478))
-   Fix tensorflow import bug ([489](https://github.com/arviz-devs/arviz/pull/489))
-   Rename N_effective to S_effective ([505](https://github.com/arviz-devs/arviz/pull/505))

### Documentation

-   Add docs to plot compare ([461](https://github.com/arviz-devs/arviz/pull/461))
-   Add InferenceData tutorial in header ([502](https://github.com/arviz-devs/arviz/pull/502))
-   Added figure to InferenceData tutorial ([510](https://github.com/arviz-devs/arviz/pull/510))

## v0.3.1 (2018 Dec 18)

### Maintenance and fixes

-   Fix installation problem with release 0.3.0

## v0.3.0 (2018 Dec 14)

-   First Beta Release
