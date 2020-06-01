# Change Log

## v0.x.x Unreleased

### New features

### Maintenance and fixes

### Deprecation

### Documentation

## v0.8.3 (2020 May 28)
### Maintenance and fixes
* Restructured internals of `from_pymc3` to handle old pymc3 releases and
  sliced traces and to provide useful warnings (#1211)

## v0.8.2 (2020 May 25)
### Maintenance and fixes
* Fixed bug in `from_pymc3` for sliced `pymc3.MultiTrace` input (#1209)

## v0.8.1 (2020 May 24)

### Maintenance and fixes
* Fixed bug in `from_pymc3` when used with PyMC3<3.9 (#1203)
* Fixed enforcement of rcParam `plot.max_subplots` in `plot_trace` and
  `plot_pair` (#1205)
* Removed extra subplot row and column in in `plot_pair` with `marginal=True` (#1205)
* Added latest PyMC3 release to CI in addition to using GitHub master (#1207)

### Documentation
* Use `dev` as version indicator in online documentation (#1204)

## v0.8.0 (2020 May 23)

### New features
* Stats and plotting functions that provide `var_names` arg can now filter parameters based on partial naming (`filter="like"`) or regular expressions (`filter="regex"`) (see [#1154](https://github.com/arviz-devs/arviz/pull/1154)).
* Add `true_values` argument for `plot_pair`. It allows for a scatter plot showing the true values of the variables #1140
* Allow xarray.Dataarray input for plots.(#1120)
* Revamped the `hpd` function to make it work with mutidimensional arrays, InferenceData and xarray objects (#1117)
* Skip test for optional/extra dependencies when not installed (#1113)
* Add option to display rank plots instead of trace (#1134)
* Add out-of-sample groups (`predictions` and `predictions_constant_data`) to `from_dict` (#1125)
* Add out-of-sample groups (`predictions` and `predictions_constant_data`) and `constant_data` group to pyro and numpyro translation (#1090, #1125)
* Add `num_chains` and `pred_dims` arguments to from_pyro and from_numpyro (#1090, #1125)
* Integrate jointplot into pairplot, add point-estimate and overlay of plot kinds (#1079)
* New grayscale style. This also add two new cmaps `cet_grey_r` and `cet_grey_r`. These are perceptually uniform gray scale cmaps from colorcet (linear_grey_10_95_c0) (#1164)
* Add warmup groups to InferenceData objects, initial support for PyStan (#1126) and PyMC3 (#1171)
* `hdi_prob` will not plot hdi if argument `hide` is passed. Previously `credible_interval` would omit HPD if `None` was passed  (#1176)
* Add `stats.ic_pointwise` rcParam (#1173)
* Add `var_name` argument to information criterion calculation: `compare`,
  `loo` and `waic` (#1173)

### Maintenance and fixes
* Fixed `plot_pair` functionality for two variables with bokeh backend (#1179)
* Changed `diagonal` argument for `marginals` and fixed `point_estimate_marker_kwargs` in `plot_pair` (#1167)
* Fixed behaviour of `credible_interval=None` in `plot_posterior` (#1115)
* Fixed hist kind of `plot_dist` with multidimensional input (#1115)
* Fixed `TypeError` in `transform` argument of `plot_density` and `plot_forest` when `InferenceData` is a list or tuple (#1121)
* Fixed overlaid pairplots issue (#1135)
* Update Docker building steps (#1127)
* Updated benchmarks and moved to asv_benchmarks/benchmarks (#1142)
* Moved `_fast_kde`, `_fast_kde_2d`, `get_bins` and `_sturges_formula` to `numeric_utils` and `get_coords` to `utils` (#1142)
* Rank plot: rename `axes` argument to `ax` (#1144)
* Added a warning specifying log scale is now the default in compare/loo/waic functions ([#1150](https://github.com/arviz-devs/arviz/pull/1150)).
* Fixed bug in `plot_posterior` with rcParam "plot.matplotlib.show" = True (#1151)
* Set `fill_last` argument of `plot_kde` to False by default (#1158)
* plot_ppc animation: improve docs and error handling (#1162)
* Fix import error when wrapped function docstring is empty (#1192)
* Fix passing axes to plot_density with several datasets ([#1198](https://github.com/arviz-devs/arviz/pull/1198))

### Deprecation
* `hpd` function deprecated in favor of `hdi`. `credible_interval` argument replaced by `hdi_prob`throughout with exception of `plot_loo_pit` (#1176)
* `plot_hpd` function deprecated in favor of `plot_hdi`. (#1190)

### Documentation
* Add classifier to `setup.py` including Matplotlib framework (#1133)
* Image thumbs generation updated to be Bokeh 2 compatible (#1116)
* Add new examples for `plot_pair` (#1110)
* Add examples for `psislw` and `r2_score` (#1129)
* Add more examples on 2D kde customization (#1158)
* Make docs compatible with sphinx3 and configure `intersphinx` for better
  references (#1184)
* Extend the developer guide and add it to the website (#1184)

## v0.7.0 (2020 Mar 2)

### New features
* Add out-of-sample predictions (`predictions` and  `predictions_constant_data` groups) to pymc3, pystan, cmdstan and cmdstanpy translations (#983, #1032 and #1064)
* Started adding pointwise log likelihood storage support (#794, #1044 and #1064)
* Add out-of-sample predictions (`predictions` and  `predictions_constant_data` groups) to pymc3 and pystan translations (#983 and #1032)
* Started adding pointwise log likelihood storage support (#794, #1044)
* Violinplot: rug-plot option (#997)
* Integrated rcParams `plot.point_estimate` (#994), `stats.ic_scale` (#993) and `stats.credible_interval` (#1017)
* Added `group` argument to `plot_ppc` (#1008), `plot_pair` (#1009) and `plot_joint` (#1012)
* Added `transform` argument to `plot_trace`, `plot_forest`, `plot_pair`, `plot_posterior`, `plot_rank`, `plot_parallel`,  `plot_violin`,`plot_density`, `plot_joint` (#1036)
* Add `skipna` argument to `hpd` and `summary` (#1035)
* Added `transform` argument to `plot_trace`, `plot_forest`, `plot_pair`, `plot_posterior`, `plot_rank`, `plot_parallel`,  `plot_violin`,`plot_density`, `plot_joint` (#1036)
* Add `marker` functionality to `bokeh_plot_elpd` (#1040)
* Add `ridgeplot_quantiles` argument to `plot_forest` (#1047)
* Added the functionality [interactive legends](https://docs.bokeh.org/en/1.4.0/docs/user_guide/interaction/legends.html) for bokeh plots of `densityplot`, `energyplot`
  and `essplot` (#1024)
* New defaults for cross validation: `loo` (old: waic) and `log` -scale (old: `deviance` -scale) (#1067)
* **Experimental Feature**: Added `arviz.wrappers` module to allow ArviZ to refit the models if necessary (#771)
* **Experimental Feature**: Added `reloo` function to ArviZ (#771)
* Added new helper function `matplotlib_kwarg_dealiaser` (#1073)
* ArviZ version to InferenceData attributes. (#1086)
* Add `log_likelihood` argument to `from_pymc3` (#1082)
* Integrated rcParams for `plot.bokeh.layout` and `plot.backend`. (#1089)
* Add automatic legends in `plot_trace` with compact=True (matplotlib only) (#1070)
* Updated hover information for `plot_pair` with bokeh backend (#1074)


### Maintenance and fixes
* Fixed bug in density and posterior plot bin computation (#1049)
* Fixed bug in density plot ax argument (#1049)
* Fixed bug in extracting prior samples for cmdstanpy (#979)
* Fix erroneous warning in traceplot (#989)
* Correct bfmi denominator (#991)
* Removed parallel from jit full (#996)
* Rename flat_inference_data_to_dict (#1003)
* Violinplot: fix histogram (#997)
* Convert all instances of SyntaxWarning to UserWarning (#1016)
* Fix `point_estimate` in `plot_posterior` (#1038)
* Fix interpolation `hpd_plot` (#1039)
* Fix `io_pymc3.py` to handle models with `potentials` (#1043)
* Fix several inconsistencies between schema and `from_pymc3` implementation
  in groups `prior`, `prior_predictive` and `observed_data` (#1045)
* Stabilize covariance matrix for `plot_kde_2d` (#1075)
* Removed extra dim in `prior` data in `from_pyro` (#1071)
* Moved CI and docs (build & deploy) to Azure Pipelines and started using codecov (#1080)
* Fixed bug in densityplot when variables differ between models (#1096)

### Deprecation
* `from_pymc3` now requires PyMC3>=3.8

### Documentation
* Updated `InferenceData` schema specification (`log_likelihood`,
  `predictions` and `predictions_constant_data` groups)
* Clarify the usage of `plot_joint` (#1001)
* Added the API link of function to examples (#1013)
* Updated PyStan_schema_example to include example of out-of-sample prediction (#1032)
* Added example for `concat` method (#1037)


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
* Add from_numpyro Integration (#811)
* Numba Google Summer of Code additions (https://ban-zee.github.io/jekyll/update/2019/08/19/Submission.html)
* Model checking, Inference Data, and Convergence assessments (https://github.com/OriolAbril/gsoc2019/blob/master/final_work_submission.md)


## v0.4.1 (2019 Jun 9)

### New features
* Reorder stats columns (#695)
* Plot Forest reports ess and rhat by default(#685)
* Add pointwise elpd (#678)

### Maintenance and fixes
* Fix io_pymc3 bug (#693)
* Fix io_pymc3 warning (#686)
* Fix 0 size bug with pystan (#677)


## v0.4.0 (2019 May 20)

### New features
* Add plot_dist (#592)
* New rhat and ess (#623)
* Add plot_hpd (#611)
* Add plot_rank (#625)

### Deprecation
* Remove load_data and save_data (#625)


## v0.3.3 (2019 Feb 23)

### New features
* Plot ppc supports multiple chains (#526)
* Plot titles now wrap (#441)
* plot_density uses a grid (#379)
* emcee reader support (#550)
* Animations in plot_ppc (#546)
* Optional dictionary for stat_funcs in summary (#583)
* Can exclude variables in selections with negated variable names (#574)

### Maintenance and fixes
* Order maintained with xarray_var_iter (#557)
* Testing very improved (multiple)
* Fix nan handling in effective sample size (#573)
* Fix kde scaling (#582)
* xticks for discrete variables (#586)
* Empty InferenceData saves consistent with netcdf (#577)
* Removes numpy pinning (#594)

### Documentation
* JOSS and Zenodo badges (#537)
* Gitter badge (#548)
* Docs for combining InferenceData (#590)

## v0.3.2 (2019 Jan 15)

### New features

* Support PyStan3 ([#464](https://github.com/arviz-devs/arviz/pull/464))
* Add some more information to the inference data of tfp ([#447](https://github.com/arviz-devs/arviz/pull/447))
* Use Split R-hat ([#477](https://github.com/arviz-devs/arviz/pull/477))
* Normalize from_xyz functions ([#490](https://github.com/arviz-devs/arviz/pull/490))
* KDE: Display quantiles ([#479](https://github.com/arviz-devs/arviz/pull/479))
* Add multiple rope support to `plot_forest` ([#448](https://github.com/arviz-devs/arviz/pull/448))
* Numba jit compilation to speed up some methods ([#515](https://github.com/arviz-devs/arviz/pull/515))
* Add `from_dict` for easier creation of az.InferenceData objects  ([#524](https://github.com/arviz-devs/arviz/pull/524))
* Add stable logsumexp ([#522](https://github.com/arviz-devs/arviz/pull/522))


### Maintenance and fixes

* Fix for `from_pyro` with multiple chains ([#463](https://github.com/arviz-devs/arviz/pull/463))
* Check `__version__` for attr ([#466](https://github.com/arviz-devs/arviz/pull/466))
* And exception to plot compare ([#461](https://github.com/arviz-devs/arviz/pull/461))
* Add Docker Testing to travisCI ([#473](https://github.com/arviz-devs/arviz/pull/473))
* fix jointplot warning ([#478](https://github.com/arviz-devs/arviz/pull/478))
* Fix tensorflow import bug ([#489](https://github.com/arviz-devs/arviz/pull/489))
* Rename N_effective to S_effective ([#505](https://github.com/arviz-devs/arviz/pull/505))


### Documentation

* Add docs to plot compare ([#461](https://github.com/arviz-devs/arviz/pull/461))
* Add InferenceData tutorial in header ([#502](https://github.com/arviz-devs/arviz/pull/502))
* Added figure to InferenceData tutorial ([#510](https://github.com/arviz-devs/arviz/pull/510))


## v0.3.1 (2018 Dec 18)

### Maintenance and fixes
* Fix installation problem with release 0.3.0

## v0.3.0 (2018 Dec 14)

* First Beta Release

