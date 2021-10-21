(doc_guide)=

## Documentation Guide

ArviZ documentation is built using a Python documentation tool, [Sphinx](https://www.sphinx-doc.org/en/master/). Sphinx converts `rst`(restructured text) files into HTML websites. There are different extensions availabel for converting other types of files into HTML websites like markdown, jupyter notebooks, etc.

Arviz [docs](https://github.com/arviz-devs/arviz/tree/main/doc/source) consist of `.rst`, `.md` and `.ipynb` files. It uses `myst-parser` and `myst-nb` for `.md` and `.ipynb` files, respectively.[Myst-parser](https://myst-parser.readthedocs.io/en/latest/sphinx/intro.html) parses all `.md` files as MyST(Markedly Structured Text).
Apart from `/doc`, ArviZ documentation also consists of docstrings. Docstrings are used in the `.py` files to explain the functions parameters and return values.

ArviZ docs also uses sphinx extensions for style, layout, navbar and putting code in the documentation. We will explore all the things one by one. Let's start!


(dev_summary)=
## Development process - summary
The preferred workflow for contributing to ArviZ is to fork
the [GitHub repository](https://github.com/arviz-devs/arviz/),
clone it to your local machine, and develop on a feature branch. the details of this process are listed on
{ref}`pr_checklist`. For a detailed
description of the recommended development process, see {ref}`building_doc_with_docker`.

## Code Formatting
For code generally follow the
[TensorFlow's style guide](https://www.tensorflow.org/community/contribute/code_style)
or the [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
Both more or less follow PEP 8.

Final formatting is done with [black](https://github.com/ambv/black).


## Docstring formatting and type hints
Docstrings should follow the
[numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).
Extra guidance can also be found in
[pandas docstring guide](https://pandas.pydata.org/pandas-docs/stable/development/contributing_docstring.html).
Please reasonably document any additions or changes to the codebase,
when in doubt, add a docstring.

The different formatting and aim between numpydoc style type description and
[type hints](https://docs.python.org/3/library/typing.html)
should be noted. numpydoc style targets docstrings and aims to be human
readable whereas type hints target function definitions and `.pyi` files and
aim to help third party tools such as type checkers or IDEs. ArviZ does not
require functions to include type hints
however contributions including them are welcome.

## Documentation for user facing methods
If changes are made to a method documented in the {ref}`ArviZ API Guide <api>`
please consider adding inline documentation examples.
You can refer to {func}`az.plot_posterior <arviz.plot_posterior>` for a good example.


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
