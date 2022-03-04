(sphinx_doc_build)=
# Building the documentation

The preferred workflow for contributing to ArviZ is to fork
the [GitHub repository](https://github.com/arviz-devs/arviz/),
clone it to your local machine, and develop on a feature branch. For a detailed description of the recommended development process, see the step-by-step guide {ref}`here <pr_steps>`.

If you are using Docker, see {ref}`building_doc_with_docker`.

## Pull request checks

Every PR has a list of checks that check if your changes follow the rules being followed by ArviZ docs.
You can check why a specific test is failing by clicking the `Details` next to it. It will take you to errors and warning page. This page shows the details of errors, for example in case of docstrings:

`arviz/plots/pairplot.py:127:0: C0301: Line too long (142/100) (line-too-long)`.

It means line 127 of the file `pairplot.py` is too long.
For running tests locally, see the {ref}`pr_checklist`.


(preview_change)=
## Previewing doc changes

There is an easy way to check the preview of docs by opening a PR on GitHub. ArviZ uses `readthedocs` to automatically build the documentation.
For previewing documentation changes, take the following steps:

1. Go to the checks of your PR. Wait for the `docs/readthedocs.org:arviz` to complete.

   ```{note}
   The green tick indicates that the given check has been built successfully.
   ```

2. Click the `Details` button next to it.
3. It will take you to the preview of ArviZ docs of your PR.
4. Go to the webpage of the file you are working on.
5. Check the preview of your changes on that page.

```{note} Note
The preview version of ArviZ docs will have a warning box that says "This page was created from a pull request (#Your PR number)." It shows the PR number whose changes have been implemented.
```

For example, a warning box will look like this:

```{warning}
This page was created from a pull request (#PR Number).
```