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


## Kwargs parameters
All the kwargs parameters are passed to the matplotlib or bokeh functions. While writing their description, the functions to which they are being passed to must be mentioned. The process for checking those functions is similar for all the kwargs arguments. Let's read the step-by-step guide for `backend_kwargs`.

### Backend Kwargs

Most of the functions contain `backend_kwargs` as an argument. It is required to add the references of functions to which they are passed as arguments. To add the related functions in `backend_kwargs` of a specific function, do the following steps:

1. Go to [arviz/plots/backends](https://github.com/arviz-devs/arviz/tree/main/arviz/plots/backends) directory. It consists of two folders, i.e.,`matplotlib` and `bokeh`.
2. Both of these directories consist of the `.py` file of your function. For example, if you are updating `plot_autocorr`, both these folders will contain `plot_autocorr.py`. We need to check the `.py` files in both directories. Let's start with `matplotlib`.
3. Go to your function file in [matplotlib directory](https://github.com/arviz-devs/arviz/tree/main/arviz/plots/backends/matplotlib).
4. Search the keyword `backend_kwargs` in the `.py` file and check which functions it is being passed to. If you hover over that function, it will turn yellow. Clicking on the function will show you the function definition points. For example, in the case of `plot_autocorr`, we will search `backend_kwrags` in [matplotlib/autocorrplot.py](https://github.com/arviz-devs/arviz/blob/main/arviz/plots/backends/matplotlib/autocorrplot.py). `backend_kwargs` are being passed to the `create_axis_grid()` function, if we click on it, we can see the two places where it is defined.
5. Go to the files where the function is defined, and search again `backend_kwargs`. For example, in the case of `plot_autocorr`, we will go to [matplotlib/__init__.py](https://github.com/arviz-devs/arviz/blob/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/matplotlib/__init__.py#L31) and [bokeh/__init__.py](https://github.com/arviz-devs/arviz/blob/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/bokeh/__init__.py#L34). If we search `backend_kwargs` here, we will see that it is being passed to `subplots()` and `figure()` in [matplotlib/__init__.py on line 55](https://github.com/arviz-devs/arviz/blob/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/matplotlib/__init__.py#L55) and [bokeh/__init__.py on line 111](https://github.com/arviz-devs/arviz/blob/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/bokeh/__init__.py#L111), respectively. 
6. Repeat steps # 4 and 5 for [backends/bokeh](https://github.com/arviz-devs/arviz/tree/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/bokeh).
7. We need to add references to the functions we got in step # 5.
8. You can go to [matplotlib documentation](https://matplotlib.org/stable/contents.html) and [bokeh documentation](https://docs.bokeh.org/en/latest/index.html) to search the functions.
9. Use the following syntax to add the function.
```
:func:`function_name`
```
For example in case of `figure()` we will use:
```
:func:`bokeh.plotting.figure`
```


## See Also
For adding the _See Also_ docstring section, you just need to add the function name. Sphinx will
automatically add links to other ArviZ objects and functions listed in the _See Also_
section.

For example, let's add :func:`~arviz.hdi` and :func:`~arviz.plot_ppc` in the _See Also_ section.

```
    See Also
    --------
    hdi : Calculate highest density interval (HDI) of array for given probability.
    plot_ppc : plot for posterior/prior predictive checks.
```