(syntax_guide)=
# Syntax Guide

## Adding targets
Adding custom targets or anchors to the headings is really helpful for cross-referencing. They allow us to link to the heading using a simple syntax.
They are defined using this syntax:

::::{tab-set}

:::{tab-item} rST
:sync: rst

```
.. _mytarget:
```
:::

:::{tab-item} MyST (Markdown)
:sync: myst

```
(mytarget)=
```
:::

::::

They are referred using this syntax:

::::{tab-set}

:::{tab-item} rST
:sync: rst

```
:ref:`mytarget`
```
:::

:::{tab-item} MyST (Markdown)
:sync: myst

```
{ref}`mytarget`
```
:::

::::

For adding anchors in `.ipynb` files, Markdown syntax will be used in the markdown cells of `.ipynb` files.

## Backticks for highlighting code keywords

For highlighting inline code keywords or file names, backticks are used. In Markdown single backticks are used while in rST double backticks are used. For example, for highlighting the file name `conf.py`, we will use this syntax:

````{tabbed} rST
```
``conf.py``
```
````
````{tabbed} MyST (Markdown)
```
`conf.py`
```
````

## Table of content tree

You might have noticed that ArviZ docs maintain a hierarchy to keep the documentation organized. A table of content tree is added to all the main pages to make the website easy to navigate. Adding the table of content tree provides the list of all pages in the left sidebar. It also enables navigation to the **Previous** and **Next** pages in the footer.

Follow this syntax for adding the table of content:

rST:
<pre>
.. toctree::
    developer_guide
    doc_guide
</pre>

MyST (Markdown):
<pre>
```{toctree}
developer_guide
doc_guide
```
</pre>

Read more about the [Sphinx toctree directive](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#table-of-contents).

(adding_references)=
## Adding references

### Hyperlinks
Complementary functions such as {func}`arviz.compare` and {func}`arviz.plot_compare` should reference
each other in their docstrings using a hyperlink, not only by name. The same
should happen with external functions whose usage is assumed to be known; a
clear example of this situation are docstrings on `kwargs` passed to bokeh or
matplotlib methods. This section covers how to reference functions from any
part of the docstring.

(reference_external_libs)=
#### Reference external libraries

Sphinx is configured to ease referencing libraries ArviZ relies heavily on by
using [intersphinx](https://docs.readthedocs.io/en/stable/guides/intersphinx.html).
See guidance on the reference about how to link to objects from external
libraries and the value of intersphinx_mapping in [conf.py](https://github.com/arviz-devs/arviz/blob/main/doc/source/conf.py) for the complete and up to date list of libraries that can be referenced.

In ArviZ docs, you can add references to functions and objects of `matplotlib`, `bokeh`, `xarray`, etc following the simple syntax. Let's try adding a function of few libraries, i.e., {meth}`xarray.Dataset.sel`, {func}`matplotlib.pyplot.subplots` and
{func}`bokeh.plotting.figure`.

````{tabbed} rST
```
:meth:`xarray.Dataset.sel`
:func:`matplotlib.pyplot.subplots`
:func:`bokeh.plotting.figure`
```
````
````{tabbed} MyST (Markdown)
```
{meth}`xarray.Dataset.sel`
{func}`matplotlib.pyplot.subplots`
{func}`bokeh.plotting.figure`
```
````

Note that the `:key:` before
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

(reference_arviz_objects)=
#### Referencing ArviZ objects

The same can be done to refer to ArviZ functions, in which case,
``:func:`arviz.loo` `` is enough, there is no need to use intersphinx.
Moreover, using ``:func:`~arviz.loo` `` will only show ``loo`` as link text
due to the preceding ``~``.

## Code tabs

You will find code tabs on every other page in ArviZ docs. As we have two main types of files, i.e, `.rst` and `.md`, we often use two code tabs to show the functionalities in both rST and Markdown.

### Synchronised Tabs
ArviZ docs are using `sphinx-design` extension for adding sync code tabs in [conf.py](https://github.com/arviz-devs/arviz/blob/main/doc/source/conf.py#L61). You can check the syntax and more info about it
at [Synchronised Tabs](https://sphinx-design.readthedocs.io/en/sbt-theme/tabs.html#synchronised-tabs). Using this extension saves us from a lot of raw-html code. Sphinx provides this extension to make our work easy and or code more concise.


## Extensions
Sphinx supports a lot of extensions to improve and customize the doc features. ArviZ makes use of these builtin and external extensions to add extra roles. See the [Extensions in ArviZ and PyMC](https://sphinx-primer.readthedocs.io/en/latest/extensions/used_extensions.html).