.. currentmodule:: arviz

rcParams
--------

.. autosummary::
    :toctree: generated/
    :template: class_no_members.rst

    rc_context

Available rcParams and Defaults
-------------------------------

Below is the list of all ArviZ rcParams, their default values, and a short description.
These values are defined in ``arvizrc.template``.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Default
     - Description

   * - data.http_protocol
     - https
     - Protocol used for loading remote datasets. Must be either ``http`` or ``https``.

   * - data.index_origin
     - 0
     - Index origin for automatically generated indices. Must be either ``0`` or ``1``.

   * - data.load
     - lazy
     - Default data loading mode. ``lazy`` uses xarray lazy loading; ``eager`` loads everything into memory.

   * - data.log_likelihood
     - true
     - Whether to save pointwise log-likelihood values. One of ``true`` or ``false``.

   * - data.metagroups
     - {...}
     - Mapping of inference groups (posterior, prior, warmup, latent, observed). See ``arvizrc.template`` for full structure.

   * - data.save_warmup
     - false
     - Whether to store warmup iterations in InferenceData.

   * - plot.backend
     - matplotlib
     - Plotting backend. One of ``matplotlib`` or ``bokeh``.

   * - plot.density_kind
     - kde
     - Density estimation method. One of ``kde`` or ``hist``.

   * - plot.max_subplots
     - 40
     - Maximum number of subplots created automatically.

   * - plot.point_estimate
     - mean
     - Point estimate shown on plots. Options are ``mean``, ``median``, ``mode``, or ``None``.

   * - plot.bokeh.bounds_x_range
     - auto
     - X-axis bounds for bokeh figures. One of ``auto``, ``None`` or a tuple of size 2.

   * - plot.bokeh.bounds_y_range
     - auto
     - Y-axis bounds for bokeh figures. One of ``auto``, ``None`` or a tuple of size 2.

   * - plot.bokeh.figure.dpi
     - 60
     - Dots-per-inch resolution for bokeh figures.

   * - plot.bokeh.figure.height
     - 500
     - Height of bokeh figures (in pixels).

   * - plot.bokeh.figure.width
     - 500
     - Width of bokeh figures (in pixels).

   * - plot.bokeh.layout.order
     - default
     - Structure of subplot layouts. One of ``default``, ``column``, ``row``, ``square``, ``square_trimmed`` or patterns like ``4row``.

   * - plot.bokeh.layout.sizing_mode
     - fixed
     - Responsive layout behavior. One of ``fixed``, ``stretch_width``, ``stretch_height``, ``stretch_both``, ``scale_width``, ``scale_height``, ``scale_both``.

   * - plot.bokeh.layout.toolbar_location
     - above
     - Toolbar position. One of ``above``, ``below``, ``left``, ``right``, or ``None`` to hide it.

   * - plot.bokeh.marker
     - cross
     - Marker type used for bokeh scatter plots.

   * - plot.bokeh.output_backend
     - webgl
     - Rendering backend. One of ``canvas``, ``svg``, ``webgl``.

   * - plot.bokeh.show
     - true
     - Whether to call ``bokeh.plotting.show``. One of ``true`` or ``false``.

   * - plot.bokeh.tools
     - reset,pan,box_zoom,wheel_zoom,lasso_select,undo,save,hover
     - Default enabled bokeh tools.

   * - plot.matplotlib.show
     - false
     - Whether to call ``plt.show`` automatically.

   * - stats.ci_prob
     - 0.94
     - Credible interval probability (e.g., similar to 95%, but ArviZ defaults to 94%).

   * - stats.information_criterion
     - loo
     - Information criterion used for model comparison. One of ``loo`` or ``waic``.

   * - stats.ic_compare_method
     - stacking
     - Method for information criterion model comparison. One of ``stacking``, ``bb-pseudo-bma``, ``pseudo-bma``.

   * - stats.ic_pointwise
     - true
     - Whether to return pointwise IC computations.

   * - stats.ic_scale
     - log
     - Scale for IC values. One of ``deviance``, ``log``, ``negative_log``.
