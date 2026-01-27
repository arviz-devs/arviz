(pr_checklist)=
# Pull request checklist

We recommend that your contribution complies with the following guidelines before you submit a pull request:

* If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

* Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress).
WIPs may be useful to (1) indicate you are working on something to avoid duplicated work,
(2) request a broad review of functionality or API, or (3) seek collaborators.

* Run any of the pre-existing examples notebooks that contain analyses that would be affected by
your changes to ensure that nothing breaks. This is a useful opportunity to not only check your
work for bugs that might not be revealed by unit test, but also to show how your contribution
improves ArviZ for end users.

* All public functions and methods must have informative NumPy-style docstrings,
including example usage when appropriate.

* New functionality must be covered by tests, and tests context must follow the
[pytest fixture pattern](https://docs.pytest.org/en/latest/fixture.html#fixture).

* Your code follows the style guidelines of the project:

  ```shell
  tox -e check
  ```

* All **tests must pass** and **coverage must not decrease**.

  ```shell
  tox -e py312-coverage
  ```

  You can replace `py312` with the version that matches your environment (e.g., `py311`, `py313`, etc.).

* Documentation can be built after the incorporation of your changes:

  ```shell
  tox -e docs
  ```