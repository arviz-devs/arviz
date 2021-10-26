(docstrings)=
# Docstring formatting and type hints

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


## Backend Kwargs



## See Also
For adding the _See Also_ docstring section, you just need to add the function name. Sphinx will
automatically add links to other ArviZ objects and functions listed in the _See Also_
section.

For example, let's add `:func:`~arviz.loo` and `:func:`~arvix.plot_ppc` in the _See Also_ section.

```
 See Also
    --------
    hdi : Calculate highest density interval (HDI) of array for given probability.
    plot_ppc : plot for posterior/prior predictive checks.
```