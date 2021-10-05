(developer_guide)=

# Developer Guide
## Library architecture
ArviZ is organized in modules (the folders in [arviz directory](https://github.com/arviz-devs/arviz/tree/main/arviz)).
The main 3 modules are `data`, `plots` and `stats`.
Then we have 3 more folders. The [tests](https://github.com/arviz-devs/arviz/tree/main/arviz/tests) folder contains tests for all these 3 modules.

The [static](https://github.com/arviz-devs/arviz/tree/main/arviz/static) folder is only used to store style and CSS files to get HTML output for `InferenceData`. Finally we have the [wrappers](https://github.com/arviz-devs/arviz/tree/main/arviz/wrappers) folder that contains experimental (not tested yet either) features and interacts closely with both [data](https://github.com/arviz-devs/arviz/tree/main/arviz/data) and [stats](https://github.com/arviz-devs/arviz/tree/main/arviz/stats) modules.

In addition, there are some files on the higher level directory: `utils.py`, `sel_utils.py`,
`rcparams.py` and `labels.py`.

## Plots
ArviZ supports multiple backends. While adding another backend, please ensure you meet the
following design patterns.

### Code Separation
Each backend should be placed in a different module per the backend.
See `arviz.plots.backends` for examples.

The code in the root level of `arviz.plots` should not contain
any opinion on backend. The idea is that the root level plotting
function performs math and constructs keywords, and the backends
code in `arviz.plots.backends` perform the backend specific
keyword argument defaulting and plot behavior.

The convenience function `get_plotting_function` available in
`arviz.plots.get_plotting_function` should be called to obtain
the correct plotting function from the associated backend. If
adding a new backend follow the pattern provided to programmatically
call the correct backend.

### Test Separation
Tests for each backend should be split into their own module
See [tests.test_plots_matplotlib](https://github.com/arviz-devs/arviz/blob/main/arviz/tests/base_tests/test_plots_matplotlib.py) for an example.

### Gallery Examples
Gallery examples are not required but encouraged. Examples are
compiled into the ArviZ documentation website. The [examples](https://github.com/arviz-devs/arviz/tree/main/examples) directory
can be found in the root of the ArviZ git repository.


## Documentation

### Docstring style
See the corresponding section in the {ref}`contributing guide <dev_summary>`.

### Hyperlinks
Complementary functions such as {func}`arviz.compare` and {func}`arviz.plot_compare` should reference
each other in their docstrings using a hyperlink, not only by name. The same
should happen with external functions whose usage is assumed to be known; a
clear example of this situation are docstrings on `kwargs` passed to bokeh or
matplotlib methods. This section covers how to reference functions from any
part of the docstring or from the _See also_ section.

#### Reference external libraries

Sphinx is configured to ease referencing libraries ArviZ relies heavily on by
using [intersphinx](https://docs.readthedocs.io/en/stable/guides/intersphinx.html).
See guidance on the reference about how to link to objects from external
libraries and the value of intersphinx_mapping in [conf.py](https://github.com/arviz-devs/arviz/blob/main/doc/source/conf.py) for the complete and up to
date list of libraries that can be referenced. Note that the `:key:` before
the reference must match the kind of object that is being referenced, it
generally will not be `:ref:` nor `:doc:`. For
example, for functions `:func:` has to be used and for class methods
`:meth:`. The complete list of keys can be found [here](https://github.com/sphinx-doc/sphinx/blob/685e3fdb49c42b464e09ec955e1033e2a8729fff/sphinx/domains/python.py#L845-L881).

The extension [sphobjinv](https://sphobjinv.readthedocs.io/en/latest/) can
also be helpful in order to get the exact type and name of a reference. Below
is an example on getting a reference from matplotlib docs:

```
$ sphobjinv suggest -t 90 -u https://matplotlib.org/objects.inv "axes.plot"

Remote inventory found.

:py:method:`matplotlib.axes.Axes.plot`
:py:method:`matplotlib.axes.Axes.plot_date`
:std:doc:`api/_as_gen/matplotlib.axes.Axes.plot`
:std:doc:`api/_as_gen/matplotlib.axes.Axes.plot_date`
```

We can therefore link to matplotlib docs on `Axes.plot` from any docstring
using:

````{tabbed} rST
```
:meth:`mpl:matplotlib.axes.Axes.plot`
```
````
````{tabbed} MyST (Markdown)
```
{meth}`mpl:matplotlib.axes.Axes.plot`
```
````


The `intersphinx_mappings`
defined for ArviZ can be seen in `conf.py`.
Moreover, the intersphinx key is optional. Thus, the pattern to get sphinx to generate links is:

````{tabbed} rST
```
:type_id:`(intersphinx_key:)object_id`
```
````
````{tabbed} MyST (Markdown)
```
{type_id}`(intersphinx_key:)object_id`
```
````

with the part between brackets being optional. See the docstring on
{meth}`~arviz.InferenceData.to_dataframe` and
[its source](https://arviz-devs.github.io/arviz/_modules/arviz/data/inference_data.html#InferenceData.to_dataframe) for an example.

#### Referencing ArviZ objects

The same can be done to refer to ArviZ functions, in which case,
``:func:`arviz.loo` `` is enough, there is no need to use intersphinx.
Moreover, using ``:func:`~arviz.loo` `` will only show ``loo`` as link text
due to the preceding ``~``.

In addition, the _See also_ docstring section is also available. Sphinx will
automatically add links to other ArviZ objects listed in the _See also_
section.
