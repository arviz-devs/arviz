(doc_dev_summary)=
# Development process - summary

The preferred workflow for contributing to ArviZ is to fork
the [GitHub repository](https://github.com/arviz-devs/arviz/),
clone it to your local machine, and develop on a feature branch. For a detailed description of the recommended development process, see the step-by-step guide {ref}`here <build_steps>`.

If you are using Docker, see {ref}`building_doc_with_docker`.

## Code Formatting
Arviz documentation conists of a lot of code snippets to demonstrate the examples. For adding code blocks, generally follow the
[TensorFlow's style guide](https://www.tensorflow.org/community/contribute/code_style)
or the [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
Both more or less follow PEP 8.

Final formatting is done with [black](https://github.com/ambv/black).
