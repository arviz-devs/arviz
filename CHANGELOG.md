# Change Log

## v0.x.x Unreleased
### New features

### Maintenance and fixes

### Deprecation

### Documentation

## v0.11.0 (2021 Dec 17)
### New features
* Added `to_dataframe` method to InferenceData ([1395](https://github.com/arviz-devs/arviz/pull/1395))
* Added `__getitem__` magic to InferenceData ([1395](https://github.com/arviz-devs/arviz/pull/1395))
* Added group argument to summary ([1408](https://github.com/arviz-devs/arviz/pull/1408))
* Add `ref_line`, `bar`, `vlines` and `marker_vlines` kwargs to `plot_rank` ([1419](https://github.com/arviz-devs/arviz/pull/1419))
* Add observed argument to (un)plot observed data in `plot_ppc` ([1422](https://github.com/arviz-devs/arviz/pull/1422))
* Add support for named dims and coordinates with multivariate observations ([1429](https://github.com/arviz-devs/arviz/pull/1429))
* Add support for discrete variables in rank plots ([1433](https://github.com/arviz-devs/arviz/pull/1433)) and
  `loo_pit` ([1500](https://github.com/arviz-devs/arviz/pull/1500))
* Add `skipna` argument to `plot_posterior` ([1432](https://github.com/arviz-devs/arviz/pull/1432))
* Make stacking the default method to compute weights in `compare` ([1438](https://github.com/arviz-devs/arviz/pull/1438))
* Add `copy()` method to `InferenceData` class. ([1501](https://github.com/arviz-devs/arviz/pull/1501)).


### Maintenance and fixes
* prevent wrapping group names in InferenceData repr_html ([1407](https://github.com/arviz-devs/arviz/pull/1407))
* Updated CmdStanPy interface ([1409](https://github.com/arviz-devs/arviz/pull/1409))
* Remove left out warning about default IC scale in `compare` ([1412](https://github.com/arviz-devs/arviz/pull/1412))
* Fixed a typo found in an error message raised in `distplot.py` ([1414](https://github.com/arviz-devs/arviz/pull/1414))
* Fix typo in `loo_pit` extraction of log likelihood ([1418](https://github.com/arviz-devs/arviz/pull/1418))
* Have `from_pystan` store attrs as strings to allow netCDF storage ([1417](https://github.com/arviz-devs/arviz/pull/1417))
* Remove ticks and spines in `plot_violin`  ([1426 ](https://github.com/arviz-devs/arviz/pull/1426))
* Use circular KDE function and fix tick labels in circular `plot_trace` ([1428](https://github.com/arviz-devs/arviz/pull/1428))
* Fix `pair_plot` for mixed discrete and continuous variables ([1434](https://github.com/arviz-devs/arviz/pull/1434))
* Fix in-sample deviance in `plot_compare` ([1435](https://github.com/arviz-devs/arviz/pull/1435))
* Fix computation of weights in compare ([1438](https://github.com/arviz-devs/arviz/pull/1438))
* Avoid repeated warning in summary ([1442](https://github.com/arviz-devs/arviz/pull/1442))
* Fix hdi failure with boolean array ([1444](https://github.com/arviz-devs/arviz/pull/1444))
* Automatically get the current axes instance for `plt_kde`, `plot_dist` and `plot_hdi` ([1452](https://github.com/arviz-devs/arviz/pull/1452))
* Add grid argument to manually specify the number of rows and columns ([1459](https://github.com/arviz-devs/arviz/pull/1459))
* Switch to `compact=True` by default in our plots ([1468](https://github.com/arviz-devs/arviz/issues/1468))
* `plot_elpd`, avoid modifying the input dict ([1477](https://github.com/arviz-devs/arviz/issues/1477))
* Do not plot divergences in `plot_trace` when `kind=rank_vlines` or `kind=rank_bars` ([1476](https://github.com/arviz-devs/arviz/issues/1476))
* Allow ignoring `observed` argument of `pymc3.DensityDist` in `from_pymc3` ([1495](https://github.com/arviz-devs/arviz/pull/1495))
* Make `from_pymc3` compatible with theano-pymc 1.1.0 ([1495](https://github.com/arviz-devs/arviz/pull/1495))
* Improve typing hints ([1491](https://github.com/arviz-devs/arviz/pull/1491), ([1492](https://github.com/arviz-devs/arviz/pull/1492),
  ([1493](https://github.com/arviz-devs/arviz/pull/1493), ([1494](https://github.com/arviz-devs/arviz/pull/1494) and
  ([1497](https://github.com/arviz-devs/arviz/pull/1497))


### Deprecation
* `plot_khat` deprecate `annotate` argument in favor of `threshold`. The new argument accepts floats ([1478](https://github.com/arviz-devs/arviz/issues/1478))

### Documentation
* Reorganize documentation and change sphinx theme ([1406](https://github.com/arviz-devs/arviz/pull/1406))
* Switch to [MyST](https://myst-parser.readthedocs.io/en/latest/) and [MyST-NB](https://myst-nb.readthedocs.io/en/latest/index.html)
  for markdown/notebook parsing in docs ([1406](https://github.com/arviz-devs/arviz/pull/1406))
* Incorporated `input_core_dims` in `hdi` and `plot_hdi` docstrings ([1410](https://github.com/arviz-devs/arviz/pull/1410))
* Add documentation pages about experimental `SamplingWrapper`s usage ([1373](https://github.com/arviz-devs/arviz/pull/1373))
* Show example titles in gallery page ([1484](https://github.com/arviz-devs/arviz/pull/1484))
* Add `sample_stats` naming convention to the InferenceData schema ([1063](https://github.com/arviz-devs/arviz/pull/1063))
* Extend api documentation about `InferenceData` methods ([1338](https://github.com/arviz-devs/arviz/pull/1338))

### Experimental
* Modified `SamplingWrapper` base API ([1373](https://github.com/arviz-devs/arviz/pull/1373))

## v0.10.0 (2020 Sep 24)
### New features
* Added InferenceData dataset containing circular variables ([1265](https://github.com/arviz-devs/arviz/pull/1265))
* Added `is_circular` argument to `plot_dist` and `plot_kde` allowing for a circular histogram (Matplotlib, Bokeh) or 1D KDE plot (Matplotlib). ([1266](https://github.com/arviz-devs/arviz/pull/1266))
* Added `to_dict` method for InferenceData object ([1223](https://github.com/arviz-devs/arviz/pull/1223))
* Added `circ_var_names` argument to `plot_trace` allowing for circular traceplot (Matplotlib) ([1336](https://github.com/arviz-devs/arviz/pull/1336))
* Ridgeplot is hdi aware. By default displays truncated densities at the specified `hdi_prop` level ([1348](https://github.com/arviz-devs/arviz/pull/1348))
* Added `plot_separation` ([1359](https://github.com/arviz-devs/arviz/pull/1359))
* Extended methods from `xr.Dataset` to `InferenceData` ([1254](https://github.com/arviz-devs/arviz/pull/1254))
* Add `extend` and `add_groups` to `InferenceData` ([1300](https://github.com/arviz-devs/arviz/pull/1300) and [1386](https://github.com/arviz-devs/arviz/pull/1386))
* Added `__iter__` method (`.items`) for InferenceData ([1356](https://github.com/arviz-devs/arviz/pull/1356))
* Add support for discrete variables in `plot_bpv` ([#1379](https://github.com/arviz-devs/arviz/pull/1379))

### Maintenance and fixes
* Automatic conversion of list/tuple to numpy array in distplot ([1277](https://github.com/arviz-devs/arviz/pull/1277))
* `plot_posterior` fix overlap of hdi and rope ([1263](https://github.com/arviz-devs/arviz/pull/1263))
* `plot_dist` bins argument error fixed ([1306](https://github.com/arviz-devs/arviz/pull/1306))
* Improve handling of circular variables in `az.summary` ([1313](https://github.com/arviz-devs/arviz/pull/1313))
* Removed change of default warning in `ELPDData` string representation ([1321](https://github.com/arviz-devs/arviz/pull/1321))
* Update `radon` example dataset to current InferenceData schema specification ([1320](https://github.com/arviz-devs/arviz/pull/1320))
* Update `from_cmdstan` functionality and add warmup groups ([1330](https://github.com/arviz-devs/arviz/pull/1330) and [1351](https://github.com/arviz-devs/arviz/pull/1351))
* Restructure plotting code to be compatible with mpl>=3.3 ([1312](https://github.com/arviz-devs/arviz/pull/1312) and [1352](https://github.com/arviz-devs/arviz/pull/1352))
* Replaced `_fast_kde()` with `kde()` which now also supports circular variables via the argument `circular` ([1284](https://github.com/arviz-devs/arviz/pull/1284)).
* Increased `from_pystan` attrs information content ([1353](https://github.com/arviz-devs/arviz/pull/1353))
* Allow `plot_trace` to return and accept axes ([1361](https://github.com/arviz-devs/arviz/pull/1361))
* Update diagnostics to be on par with posterior package ([1366](https://github.com/arviz-devs/arviz/pull/1366))
* Use method="average" in `scipy.stats.rankdata` ([1380](https://github.com/arviz-devs/arviz/pull/1380))
* Add more `plot_parallel` examples ([1380](https://github.com/arviz-devs/arviz/pull/1380))
* Bump minimum xarray version to 0.16.1 ([1389](https://github.com/arviz-devs/arviz/pull/1389)
* Fix multi rope for `plot_forest` ([1390](https://github.com/arviz-devs/arviz/pull/1390))
* Bump minimum xarray version to 0.16.1 ([1389](https://github.com/arviz-devs/arviz/pull/1389))
* `from_dict` will now store warmup groups even with the main group missing ([1386](https://github.com/arviz-devs/arviz/pull/1386))
* increase robustness for repr_html handling ([1392](https://github.com/arviz-devs/arviz/pull/1392))

## v0.9.0 (2020 June 23)
### New features
* loo-pit plot. The kde is computed over the data interval (this could be shorter than [0, 1]). The HDI is computed analitically ([1215](https://github.com/arviz-devs/arviz/pull/1215))
* Added `html_repr` of InferenceData objects for jupyter notebooks. ([1217](https://github.com/arviz-devs/arviz/pull/1217))
* Added support for PyJAGS via the function `from_pyjags`. ([1219](https://github.com/arviz-devs/arviz/pull/1219) and [1245](https://github.com/arviz-devs/arviz/pull/1245))
* `from_pymc3` can now retrieve `coords` and `dims` from model context ([1228](https://github.com/arviz-devs/arviz/pull/1228), [1240](https://github.com/arviz-devs/arviz/pull/1240) and [1249](https://github.com/arviz-devs/arviz/pull/1249))
* `plot_trace` now supports multiple aesthetics to identify chain and variable
  shape and support matplotlib aliases ([1253](https://github.com/arviz-devs/arviz/pull/1253))
* `plot_hdi` can now take already computed HDI values ([1241](https://github.com/arviz-devs/arviz/pull/1241))
* `plot_bpv`. A new plot for Bayesian p-values ([1222](https://github.com/arviz-devs/arviz/pull/1222))

### Maintenance and fixes
* Include data from `MultiObservedRV` to `observed_data` when using
  `from_pymc3` ([1098](https://github.com/arviz-devs/arviz/pull/1098))
* Added a note on `plot_pair` when trying to use `plot_kde` on `InferenceData`
  objects. ([1218](https://github.com/arviz-devs/arviz/pull/1218))
* Added `log_likelihood` argument to `from_pyro` and a warning if log likelihood cannot be obtained ([1227](https://github.com/arviz-devs/arviz/pull/1227))
* Skip tests on matplotlib animations if ffmpeg is not installed ([1227](https://github.com/arviz-devs/arviz/pull/1227))
* Fix hpd bug where arguments were being ignored ([1236](https://github.com/arviz-devs/arviz/pull/1236))
* Remove false positive warning in `plot_hdi` and fixed matplotlib axes generation ([1241](https://github.com/arviz-devs/arviz/pull/1241))
* Change the default `zorder` of scatter points from `0` to `0.6` in `plot_pair` ([1246](https://github.com/arviz-devs/arviz/pull/1246))
* Update `get_bins` for numpy 1.19 compatibility ([1256](https://github.com/arviz-devs/arviz/pull/1256))
* Fixes to `rug`, `divergences` arguments in `plot_trace` ([1253](https://github.com/arviz-devs/arviz/pull/1253))

### Deprecation
* Using `from_pymc3` without a model context available now raises a
  `FutureWarning` and will be deprecated in a future version ([1227](https://github.com/arviz-devs/arviz/pull/1227))
* In `plot_trace`, `chain_prop` and `compact_prop` as tuples will now raise a
  `FutureWarning` ([1253](https://github.com/arviz-devs/arviz/pull/1253))
* `hdi` with 2d data raises a FutureWarning ([1241](https://github.com/arviz-devs/arviz/pull/1241))

### Documentation
* A section has been added to the documentation at InferenceDataCookbook.ipynb illustrating the use of ArviZ in conjunction with PyJAGS. ([1219](https://github.com/arviz-devs/arviz/pull/1219) and [1245](https://github.com/arviz-devs/arviz/pull/1245))
* Fixed inconsistent capitalization in `plot_hdi` docstring ([1221](https://github.com/arviz-devs/arviz/pull/1221))
* Fixed and extended `InferenceData.map` docs ([1255](https://github.com/arviz-devs/arviz/pull/1255))

## v0.8.3 (2020 May 28)
### Maintenance and fixes
* Restructured internals of `from_pymc3` to handle old pymc3 releases and
  sliced traces and to provide useful warnings ([1211](https://github.com/arviz-devs/arviz/pull/1211))


## v0.8.2 (2020 May 25)
### Maintenance and fixes
* Fixed bug in `from_pymc3` for sliced `pymc3.MultiTrace` input ([1209](https://github.com/arviz-devs/arviz/pull/1209))

## v0.8.1 (2020 May 24)

### Maintenance and fixes
* Fixed bug in `from_pymc3` when used with PyMC3<3.9 ([1203](https://github.com/arviz-devs/arviz/pull/1203))
* Fixed enforcement of rcParam `plot.max_subplots` in `plot_trace` and
  `plot_pair` ([1205](https://github.com/arviz-devs/arviz/pull/1205))
* Removed extra subplot row and column in in `plot_pair` with `marginal=True` ([1205](https://github.com/arviz-devs/arviz/pull/1205))
* Added latest PyMC3 release to CI in addition to using GitHub master ([1207](https://github.com/arviz-devs/arviz/pull/1207))

### Documentation
* Use `dev` as version indicator in online documentation ([1204](https://github.com/arviz-devs/arviz/pull/1204))

## v0.8.0 (2020 May 23)

### New features
* Stats and plotting functions that provide `var_names` arg can now filter parameters based on partial naming (`filter="like"`) or regular expressions (`filter="regex"`) (see [1154](https://github.com/arviz-devs/arviz/pull/1154)).
* Add `true_values` argument for `plot_pair`. It allows for a scatter plot showing the true values of the variables ([1140](https://github.com/arviz-devs/arviz/pull/1140))
* Allow xarray.Dataarray input for plots.([1120](https://github.com/arviz-devs/arviz/pull/1120))
* Revamped the `hpd` function to make it work with mutidimensional arrays, InferenceData and xarray objects ([1117](https://github.com/arviz-devs/arviz/pull/1117))
* Skip test for optional/extra dependencies when not installed ([1113](https://github.com/arviz-devs/arviz/pull/1113))
* Add option to display rank plots instead of trace ([1134](https://github.com/arviz-devs/arviz/pull/1134))
* Add out-of-sample groups (`predictions` and `predictions_constant_data`) to `from_dict` ([1125](https://github.com/arviz-devs/arviz/pull/1125))
* Add out-of-sample groups (`predictions` and `predictions_constant_data`) and `constant_data` group to pyro and numpyro translation ([1090](https://github.com/arviz-devs/arviz/pull/1090), [1125](https://github.com/arviz-devs/arviz/pull/1125))
* Add `num_chains` and `pred_dims` arguments to from_pyro and from_numpyro ([1090](https://github.com/arviz-devs/arviz/pull/1090), [1125](https://github.com/arviz-devs/arviz/pull/1125))
* Integrate jointplot into pairplot, add point-estimate and overlay of plot kinds ([1079](https://github.com/arviz-devs/arviz/pull/1079))
* New grayscale style. This also add two new cmaps `cet_grey_r` and `cet_grey_r`. These are perceptually uniform gray scale cmaps from colorcet (linear_grey_10_95_c0) ([1164](https://github.com/arviz-devs/arviz/pull/1164))
* Add warmup groups to InferenceData objects, initial support for PyStan ([1126](https://github.com/arviz-devs/arviz/pull/1126)) and PyMC3 ([1171](https://github.com/arviz-devs/arviz/pull/1171))
* `hdi_prob` will not plot hdi if argument `hide` is passed. Previously `credible_interval` would omit HPD if `None` was passed  ([1176](https://github.com/arviz-devs/arviz/pull/1176))
* Add `stats.ic_pointwise` rcParam ([1173](https://github.com/arviz-devs/arviz/pull/1173))
* Add `var_name` argument to information criterion calculation: `compare`,
  `loo` and `waic` ([1173](https://github.com/arviz-devs/arviz/pull/1173))

### Maintenance and fixes
* Fixed `plot_pair` functionality for two variables with bokeh backend ([1179](https://github.com/arviz-devs/arviz/pull/1179))
* Changed `diagonal` argument for `marginals` and fixed `point_estimate_marker_kwargs` in `plot_pair` ([1167](https://github.com/arviz-devs/arviz/pull/1167))
* Fixed behaviour of `credible_interval=None` in `plot_posterior` ([1115](https://github.com/arviz-devs/arviz/pull/1115))
* Fixed hist kind of `plot_dist` with multidimensional input ([1115](https://github.com/arviz-devs/arviz/pull/1115))
* Fixed `TypeError` in `transform` argument of `plot_density` and `plot_forest` when `InferenceData` is a list or tuple ([1121](https://github.com/arviz-devs/arviz/pull/1121))
* Fixed overlaid pairplots issue ([1135](https://github.com/arviz-devs/arviz/pull/1135))
* Update Docker building steps ([1127](https://github.com/arviz-devs/arviz/pull/1127))
* Updated benchmarks and moved to asv_benchmarks/benchmarks ([1142](https://github.com/arviz-devs/arviz/pull/1142))
* Moved `_fast_kde`, `_fast_kde_2d`, `get_bins` and `_sturges_formula` to `numeric_utils` and `get_coords` to `utils` ([1142](https://github.com/arviz-devs/arviz/pull/1142))
* Rank plot: rename `axes` argument to `ax` ([1144](https://github.com/arviz-devs/arviz/pull/1144))
* Added a warning specifying log scale is now the default in compare/loo/waic functions ([1150](https://github.com/arviz-devs/arviz/pull/1150)).
* Fixed bug in `plot_posterior` with rcParam "plot.matplotlib.show" = True ([1151](https://github.com/arviz-devs/arviz/pull/1151))
* Set `fill_last` argument of `plot_kde` to False by default ([1158](https://github.com/arviz-devs/arviz/pull/1158))
* plot_ppc animation: improve docs and error handling ([1162](https://github.com/arviz-devs/arviz/pull/1162))
* Fix import error when wrapped function docstring is empty ([1192](https://github.com/arviz-devs/arviz/pull/1192))
* Fix passing axes to plot_density with several datasets ([1198](https://github.com/arviz-devs/arviz/pull/1198))

### Deprecation
* `hpd` function deprecated in favor of `hdi`. `credible_interval` argument replaced by `hdi_prob`throughout with exception of `plot_loo_pit` ([1176](https://github.com/arviz-devs/arviz/pull/1176))
* `plot_hpd` function deprecated in favor of `plot_hdi`. ([1190](https://github.com/arviz-devs/arviz/pull/1190))

### Documentation
* Add classifier to `setup.py` including Matplotlib framework ([1133](https://github.com/arviz-devs/arviz/pull/1133))
* Image thumbs generation updated to be Bokeh 2 compatible ([1116](https://github.com/arviz-devs/arviz/pull/1116))
* Add new examples for `plot_pair` ([1110](https://github.com/arviz-devs/arviz/pull/1110))
* Add examples for `psislw` and `r2_score` ([1129](https://github.com/arviz-devs/arviz/pull/1129))
* Add more examples on 2D kde customization ([1158](https://github.com/arviz-devs/arviz/pull/1158))
* Make docs compatible with sphinx3 and configure `intersphinx` for better
  references ([1184](https://github.com/arviz-devs/arviz/pull/1184))
* Extend the developer guide and add it to the website ([1184](https://github.com/arviz-devs/arviz/pull/1184))

## v0.7.0 (2020 Mar 2)

### New features
* Add out-of-sample predictions (`predictions` and  `predictions_constant_data` groups) to pymc3, pystan, cmdstan and cmdstanpy translations ([983](https://github.com/arviz-devs/arviz/pull/983), [1032](https://github.com/arviz-devs/arviz/pull/1032) and [1064](https://github.com/arviz-devs/arviz/pull/1064))
* Started adding pointwise log likelihood storage support ([794](https://github.com/arviz-devs/arviz/pull/794), [1044](https://github.com/arviz-devs/arviz/pull/1044) and [1064](https://github.com/arviz-devs/arviz/pull/1064))
* Add out-of-sample predictions (`predictions` and  `predictions_constant_data` groups) to pymc3 and pystan translations ([983](https://github.com/arviz-devs/arviz/pull/983) and [1032](https://github.com/arviz-devs/arviz/pull/1032))
* Started adding pointwise log likelihood storage support ([794](https://github.com/arviz-devs/arviz/pull/794), [1044](https://github.com/arviz-devs/arviz/pull/1044))
* Violinplot: rug-plot option ([997](https://github.com/arviz-devs/arviz/pull/997))
* Integrated rcParams `plot.point_estimate` ([994](https://github.com/arviz-devs/arviz/pull/994)), `stats.ic_scale` ([993](https://github.com/arviz-devs/arviz/pull/993)) and `stats.credible_interval` ([1017](https://github.com/arviz-devs/arviz/pull/1017))
* Added `group` argument to `plot_ppc` ([1008](https://github.com/arviz-devs/arviz/pull/1008)), `plot_pair` ([1009](https://github.com/arviz-devs/arviz/pull/1009)) and `plot_joint` ([1012](https://github.com/arviz-devs/arviz/pull/1012))
* Added `transform` argument to `plot_trace`, `plot_forest`, `plot_pair`, `plot_posterior`, `plot_rank`, `plot_parallel`,  `plot_violin`,`plot_density`, `plot_joint` ([1036](https://github.com/arviz-devs/arviz/pull/1036))
* Add `skipna` argument to `hpd` and `summary` ([1035](https://github.com/arviz-devs/arviz/pull/1035))
* Added `transform` argument to `plot_trace`, `plot_forest`, `plot_pair`, `plot_posterior`, `plot_rank`, `plot_parallel`,  `plot_violin`,`plot_density`, `plot_joint` ([1036](https://github.com/arviz-devs/arviz/pull/1036))
* Add `marker` functionality to `bokeh_plot_elpd` ([1040](https://github.com/arviz-devs/arviz/pull/1040))
* Add `ridgeplot_quantiles` argument to `plot_forest` ([1047](https://github.com/arviz-devs/arviz/pull/1047))
* Added the functionality [interactive legends](https://docs.bokeh.org/en/1.4.0/docs/user_guide/interaction/legends.html) for bokeh plots of `densityplot`, `energyplot`
  and `essplot` ([1024](https://github.com/arviz-devs/arviz/pull/1024))
* New defaults for cross validation: `loo` (old: waic) and `log` -scale (old: `deviance` -scale) ([1067](https://github.com/arviz-devs/arviz/pull/1067))
* **Experimental Feature**: Added `arviz.wrappers` module to allow ArviZ to refit the models if necessary ([771](https://github.com/arviz-devs/arviz/pull/771))
* **Experimental Feature**: Added `reloo` function to ArviZ ([771](https://github.com/arviz-devs/arviz/pull/771))
* Added new helper function `matplotlib_kwarg_dealiaser` ([1073](https://github.com/arviz-devs/arviz/pull/1073))
* ArviZ version to InferenceData attributes. ([1086](https://github.com/arviz-devs/arviz/pull/1086))
* Add `log_likelihood` argument to `from_pymc3` ([1082](https://github.com/arviz-devs/arviz/pull/1082))
* Integrated rcParams for `plot.bokeh.layout` and `plot.backend`. ([1089](https://github.com/arviz-devs/arviz/pull/1089))
* Add automatic legends in `plot_trace` with compact=True (matplotlib only) ([1070](https://github.com/arviz-devs/arviz/pull/1070))
* Updated hover information for `plot_pair` with bokeh backend ([1074](https://github.com/arviz-devs/arviz/pull/1074))


### Maintenance and fixes
* Fixed bug in density and posterior plot bin computation ([1049](https://github.com/arviz-devs/arviz/pull/1049))
* Fixed bug in density plot ax argument ([1049](https://github.com/arviz-devs/arviz/pull/1049))
* Fixed bug in extracting prior samples for cmdstanpy ([979](https://github.com/arviz-devs/arviz/pull/979))
* Fix erroneous warning in traceplot ([989](https://github.com/arviz-devs/arviz/pull/989))
* Correct bfmi denominator ([991](https://github.com/arviz-devs/arviz/pull/991))
* Removed parallel from jit full ([996](https://github.com/arviz-devs/arviz/pull/996))
* Rename flat_inference_data_to_dict ([1003](https://github.com/arviz-devs/arviz/pull/1003))
* Violinplot: fix histogram ([997](https://github.com/arviz-devs/arviz/pull/997))
* Convert all instances of SyntaxWarning to UserWarning ([1016](https://github.com/arviz-devs/arviz/pull/1016))
* Fix `point_estimate` in `plot_posterior` ([1038](https://github.com/arviz-devs/arviz/pull/1038))
* Fix interpolation `hpd_plot` ([1039](https://github.com/arviz-devs/arviz/pull/1039))
* Fix `io_pymc3.py` to handle models with `potentials` ([1043](https://github.com/arviz-devs/arviz/pull/1043))
* Fix several inconsistencies between schema and `from_pymc3` implementation
  in groups `prior`, `prior_predictive` and `observed_data` ([1045](https://github.com/arviz-devs/arviz/pull/1045))
* Stabilize covariance matrix for `plot_kde_2d` ([1075](https://github.com/arviz-devs/arviz/pull/1075))
* Removed extra dim in `prior` data in `from_pyro` ([1071](https://github.com/arviz-devs/arviz/pull/1071))
* Moved CI and docs (build & deploy) to Azure Pipelines and started using codecov ([1080](https://github.com/arviz-devs/arviz/pull/1080))
* Fixed bug in densityplot when variables differ between models ([1096](https://github.com/arviz-devs/arviz/pull/1096))

### Deprecation
* `from_pymc3` now requires PyMC3>=3.8

### Documentation
* Updated `InferenceData` schema specification (`log_likelihood`,
  `predictions` and `predictions_constant_data` groups)
* Clarify the usage of `plot_joint` ([1001](https://github.com/arviz-devs/arviz/pull/1001))
* Added the API link of function to examples ([1013](https://github.com/arviz-devs/arviz/pull/1013))
* Updated PyStan_schema_example to include example of out-of-sample prediction ([1032](https://github.com/arviz-devs/arviz/pull/1032))
* Added example for `concat` method ([1037](https://github.com/arviz-devs/arviz/pull/1037))


## v0.6.1 (2019 Dec 28)

### New features
* Update for pair_plot divergences can be selected
* Default tools follow global (ArviZ) defaults
* Add interactive legend for a plot, if only two variables are used in pairplot

### Maintenance and fixes
* Change `packaging` import from absolute to relative format, explicitly importing `version` function


## v0.6.0 (2019 Dec 24)

### New features

* Initial bokeh support.
* ArviZ.jl a Julia interface to ArviZ (@sethaxen )

### Maintenance and fixes

* Fully support `numpyro` (@fehiepsi )
* log_likelihood and observed data from pyro
* improve rcparams
* fix `az.concat` functionality (@anzelpwj )

### Documentation
* distplot docstring plotting example (@jscarbor )

## v0.5.1 (2019 Sep 16)

### Maintenance and fixes
* Comment dev requirements in setup.py


## v0.5.0 (2019 Sep 15)

## New features
* Add from_numpyro Integration ([811](https://github.com/arviz-devs/arviz/pull/811))
* Numba Google Summer of Code additions (https://ban-zee.github.io/jekyll/update/2019/08/19/Submission.html)
* Model checking, Inference Data, and Convergence assessments (https://github.com/OriolAbril/gsoc2019/blob/master/final_work_submission.md)


## v0.4.1 (2019 Jun 9)

### New features
* Reorder stats columns ([695](https://github.com/arviz-devs/arviz/pull/695))
* Plot Forest reports ess and rhat by default([685](https://github.com/arviz-devs/arviz/pull/685))
* Add pointwise elpd ([678](https://github.com/arviz-devs/arviz/pull/678))

### Maintenance and fixes
* Fix io_pymc3 bug ([693](https://github.com/arviz-devs/arviz/pull/693))
* Fix io_pymc3 warning ([686](https://github.com/arviz-devs/arviz/pull/686))
* Fix 0 size bug with pystan ([677](https://github.com/arviz-devs/arviz/pull/677))


## v0.4.0 (2019 May 20)

### New features
* Add plot_dist ([592](https://github.com/arviz-devs/arviz/pull/592))
* New rhat and ess ([623](https://github.com/arviz-devs/arviz/pull/623))
* Add plot_hpd ([611](https://github.com/arviz-devs/arviz/pull/611))
* Add plot_rank ([625](https://github.com/arviz-devs/arviz/pull/625))

### Deprecation
* Remove load_data and save_data ([625](https://github.com/arviz-devs/arviz/pull/625))


## v0.3.3 (2019 Feb 23)

### New features
* Plot ppc supports multiple chains ([526](https://github.com/arviz-devs/arviz/pull/526))
* Plot titles now wrap ([441](https://github.com/arviz-devs/arviz/pull/441))
* plot_density uses a grid ([379](https://github.com/arviz-devs/arviz/pull/379))
* emcee reader support ([550](https://github.com/arviz-devs/arviz/pull/550))
* Animations in plot_ppc ([546](https://github.com/arviz-devs/arviz/pull/546))
* Optional dictionary for stat_funcs in summary ([583](https://github.com/arviz-devs/arviz/pull/583))
* Can exclude variables in selections with negated variable names ([574](https://github.com/arviz-devs/arviz/pull/574))

### Maintenance and fixes
* Order maintained with xarray_var_iter ([557](https://github.com/arviz-devs/arviz/pull/557))
* Testing very improved (multiple)
* Fix nan handling in effective sample size ([573](https://github.com/arviz-devs/arviz/pull/573))
* Fix kde scaling ([582](https://github.com/arviz-devs/arviz/pull/582))
* xticks for discrete variables ([586](https://github.com/arviz-devs/arviz/pull/586))
* Empty InferenceData saves consistent with netcdf ([577](https://github.com/arviz-devs/arviz/pull/577))
* Removes numpy pinning ([594](https://github.com/arviz-devs/arviz/pull/594))

### Documentation
* JOSS and Zenodo badges ([537](https://github.com/arviz-devs/arviz/pull/537))
* Gitter badge ([548](https://github.com/arviz-devs/arviz/pull/548))
* Docs for combining InferenceData ([590](https://github.com/arviz-devs/arviz/pull/590))

## v0.3.2 (2019 Jan 15)

### New features

* Support PyStan3 ([464](https://github.com/arviz-devs/arviz/pull/464))
* Add some more information to the inference data of tfp ([447](https://github.com/arviz-devs/arviz/pull/447))
* Use Split R-hat ([477](https://github.com/arviz-devs/arviz/pull/477))
* Normalize from_xyz functions ([490](https://github.com/arviz-devs/arviz/pull/490))
* KDE: Display quantiles ([479](https://github.com/arviz-devs/arviz/pull/479))
* Add multiple rope support to `plot_forest` ([448](https://github.com/arviz-devs/arviz/pull/448))
* Numba jit compilation to speed up some methods ([515](https://github.com/arviz-devs/arviz/pull/515))
* Add `from_dict` for easier creation of az.InferenceData objects  ([524](https://github.com/arviz-devs/arviz/pull/524))
* Add stable logsumexp ([522](https://github.com/arviz-devs/arviz/pull/522))


### Maintenance and fixes

* Fix for `from_pyro` with multiple chains ([463](https://github.com/arviz-devs/arviz/pull/463))
* Check `__version__` for attr ([466](https://github.com/arviz-devs/arviz/pull/466))
* And exception to plot compare ([461](https://github.com/arviz-devs/arviz/pull/461))
* Add Docker Testing to travisCI ([473](https://github.com/arviz-devs/arviz/pull/473))
* fix jointplot warning ([478](https://github.com/arviz-devs/arviz/pull/478))
* Fix tensorflow import bug ([489](https://github.com/arviz-devs/arviz/pull/489))
* Rename N_effective to S_effective ([505](https://github.com/arviz-devs/arviz/pull/505))


### Documentation

* Add docs to plot compare ([461](https://github.com/arviz-devs/arviz/pull/461))
* Add InferenceData tutorial in header ([502](https://github.com/arviz-devs/arviz/pull/502))
* Added figure to InferenceData tutorial ([510](https://github.com/arviz-devs/arviz/pull/510))


## v0.3.1 (2018 Dec 18)

### Maintenance and fixes
* Fix installation problem with release 0.3.0

## v0.3.0 (2018 Dec 14)

* First Beta Release
