(pr_checklist)=
# Pull request checklist

We recommend that your contribution complies with the following guidelines before you submit a pull request:

* If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

* Please use a Draft Pull Request to indicate an incomplete contribution or work in progress.
Draft PRs may be useful to (1) signal you are working on something and avoid duplicated work,
(2) request early feedback on functionality or API, or (3) seek collaborators.

* Run any of the pre-existing examples notebooks that contain analyses that would be affected by
your changes to ensure that nothing breaks. This is a useful opportunity to not only check your
work for bugs that might not be revealed by unit test, but also to show how your contribution
improves ArviZ for end users.

* All public functions and methods must have informative NumPy-style docstrings,
including Examples and, when appropriate, See Also, References, and Notes.
You can refer to {func}`~arviz_plots.plot_dist` and {func}`~arviz_plots.plot_energy` for good examples.

* New functionality must be covered by tests, and tests context must follow the
[pytest fixture pattern](https://docs.pytest.org/en/latest/fixture.html#fixture).

* Your code follows the style guidelines of the project:

  ```shell
  tox -e check
  ```

* All **tests must pass**.

  ```shell
  tox -e py312
  ```

  You can replace `py312` with the version that matches your environment (e.g., `py311`, `py313`, etc.).

  Moreover, given that libraries have their own testing tasks, it's recommended to check their specific
  testing guidelines:

  * ArviZ-base: coming soon
  * {ref}`ArviZ-stats <arviz_stats:contributing_testing>`
  * {ref}`ArviZ-plots <arviz_plots:contributing_testing>`

* Documentation can be built after the incorporation of your changes:

  ```shell
  tox -e docs
  ```