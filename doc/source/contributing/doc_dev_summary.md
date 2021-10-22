(doc_dev_summary)=
# Development process - summary

The preferred workflow for contributing to ArviZ is to fork
the [GitHub repository](https://github.com/arviz-devs/arviz/),
clone it to your local machine, and develop on a feature branch. For a detailed description of the recommended development process, see the step-by-step guide {ref}`here <build_steps>`.

If you are using Docker, see {ref}`building_doc_with_docker`.

## Code Formatting
Arviz documentation consists of a lot of code snippets to demonstrate the examples. For adding code blocks, generally follow the [TensorFlow's style guide](https://www.tensorflow.org/community/contribute/code_style)
or the [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
Both more or less follow PEP 8.

Final formatting is done with [black](https://github.com/ambv/black).

## Previewing doc changes

There is an easy way to check the preview of docs by using GitHub. Open the PR and GitHub actions will automatically build the documentation.
For previewing documentation changes, take the following steps:

1. Go to the checks of your PR. Wait for the `docs/readthedocs.org:arviz` to complete.

   ```{note}
   The green tick indicates that the given check has been built successfully.
   ```

2. Click the `Details` button next to it.
3. It will take to you the preview of ArviZ docs of your PR.
4. Go to the webpage of the file you are working on.
5. Check the preview of your changes on that page.

```{note} Note
The preview version of ArviZ docs will have a warning box that says "This page was created from a pull request (#Your PR number)." It shows the PR number whose changes have been implemented.
```

For example, a warning box will look like this:

```{warning}
This page was created from a pull request (#PR Number).
```