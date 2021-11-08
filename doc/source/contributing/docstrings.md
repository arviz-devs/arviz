(docstrings)=
# Docstrings

Docstrings should follow the
[numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html). Read more about it {ref}`docstring_formatting`.
In ArviZ docs, docstrings consit of six main sections, i.e.,
1. Short summary
2. Extended Summary
3. Parameters
4. Returns
5. See Also
6. References
7. Examples

## References
While adding description of parameters, examples, etc, it is important to add refernces to external libraries and ArviZ.
Docstrings follow the same guide for adding references as the other docs.
For adding references to external libraries functions and objects, see {ref}`reference_external_libs`. For referencing ArviZ objects, follow {ref}`reference_arviz_objects`.

hat must be present because we have a lot of interconnected functions both within the library and with external libraries like xarray for example and we have seen that users have a hard time finding the functions they are searching for. But I would not focus too much (or at all) on how to format it as numpydoc already does that, we could for example explain things that are not covered in the numpydoc style guide like how both arviz.plot_hdi and plot_hdi work and link to the same place but show different text for the hyperlink

## See Also
In ArviZ docs, we have a lot of interconnected functions both within the library and with external libraries and it can take a lot of time to search for the related functions. It is cruical to add the See Also section to save users time.
For adding the _See Also_ docstring section, you just need to add the function name and its short summary. Sphinx will
automatically add links to other ArviZ objects and functions listed in the _See Also_
section.

For example, let's add {func}`~arviz.hdi` and {func}`~arviz.plot_ppc` in the _See Also_ section.

```
    See Also
    --------
    hdi : Calculate highest density interval (HDI) of array for given probability.
    plot_ppc : plot for posterior/prior predictive checks.
```

## Kwargs parameters
All the kwargs parameters in {ref}`plots <plot_api>` modules are passed to the matplotlib or bokeh functions. While writing their description, the functions to which they are being passed must be mentioned. In order to check or add those functions, the process is the same for all the kwargs arguments. Let's read the step-by-step guide for `backend_kwargs` as an example.

### Backend Kwargs

Take the following steps to add the related functions:

1. Go to [arviz/plots/backends](https://github.com/arviz-devs/arviz/tree/main/arviz/plots/backends) directory. It consists of two folders, i.e., `matplotlib` and `bokeh`.

2. If you go to these directories, you will notice they consist of the `.py` file of all the functions. For example, if you are updating `plot_autocorr`, both these folders will contain `plot_autocorr.py`. We need to check the `.py` files in both directories. Let's start with `matplotlib`.

3. Go to your function file in [matplotlib directory](https://github.com/arviz-devs/arviz/tree/main/arviz/plots/backends/matplotlib).

4. First of all, search the keyword `backend_kwargs` in the `.py` file. Now, check the function to which it is being passed as an argument. If you hover over the function, it will turn yellow and clicking on the function will take you to the function definition points.
For example, in the case of `plot_autocorr`, we will search `backend_kwrags` in [matplotlib/autocorrplot.py](https://github.com/arviz-devs/arviz/blob/main/arviz/plots/backends/matplotlib/autocorrplot.py). `backend_kwargs` are being passed to the `create_axis_grid()` function on [line 46](https://github.com/arviz-devs/arviz/blob/main/arviz/plots/backends/matplotlib/autocorrplot.py#L46), if we click on `create_axes_grid`, we can see the two places where it is defined, i.e., `bokeh/__init__.py` and `matplotlib/__init__.py`.

5. Go to the files where the function is defined one by one, and search again `backend_kwargs`. For example, in case of `plot_autocorr`, we will go to [matplotlib/__init__.py](https://github.com/arviz-devs/arviz/blob/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/matplotlib/__init__.py#L31) and [bokeh/__init__.py](https://github.com/arviz-devs/arviz/blob/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/bokeh/__init__.py#L34). If we search `backend_kwargs` in these files, we will see that it is being passed to `subplots()` and `figure()` in [matplotlib/__init__.py on line 55](https://github.com/arviz-devs/arviz/blob/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/matplotlib/__init__.py#L55) and [bokeh/__init__.py on line 111](https://github.com/arviz-devs/arviz/blob/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/bokeh/__init__.py#L111), respectively.

6. Repeat steps # 4 and 5 for [backends/bokeh](https://github.com/arviz-devs/arviz/tree/a934308e8d8f63b2b6b06b3badf7c93a88112c97/arviz/plots/backends/bokeh).

7. Now, we got the related functions. The next step is to add references to these functions.
   Use the following syntax to add the function.
   ```
   :func:`function_name`
   ```
   For example, the code for adding {meth}`xarray.Dataset.sel`, {func}`matplotlib.pyplot.subplots` and
   {func}`bokeh.plotting.figure` is given below.
   ```
   :meth:`xarray.Dataset.sel`, :func:`matplotlib.pyplot.subplots` and :func:`bokeh.plotting.figure`
   ```

   You can also go to [matplotlib documentation](https://matplotlib.org/stable/contents.html) and [bokeh documentation](https://docs.bokeh.org/en/latest/index.html) to search these functions.
