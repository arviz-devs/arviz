# How to make a new release

ArviZ uses the following process to cut a new release of the ArviZverse packages.

1. Bump the version number in `src/<arviz repo>/__init__.py` or `src/<arviz repo>/_version.py`

   ```diff
   - __version = "1.2.0"
   + __version = "1.2.1"
   ```

2. Check versions in `pyproject.toml`. Arviz aims to follow the recommendations in [SPEC-0](https://scientific-python.org/specs/spec-0000/) from scientific python.

3. Open a Pull Request including these changes. Make sure all CI tests pass, adding commits if necessary.
   Even if CI is passing on main, there might be new releases of dependencies that break CI.

   :::{important}
   The documentation build should also complete successfully. If we publish a release from a commit
   where the docs don't build we then have to choose between that version not showing in the
   version switcher or publishing a patch release that fixes the doc build.
   :::

4. Add a release in the Github [release page](https://github.com/arviz-devs/arviz/releases) once the PR is merged.

5. After the release on Github, the CI system will complete the rest of the steps. Including making any wheels and uploading the new version to PyPI.
   It will also add a PR to update the changelog.

6. Add a follow-up PR changing the version string to include the dev flag.
   Make sure the version string is [PEP 440](https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions) compliant.
   For example, after releasing `v0.12.1` it should be set to `0.13.0.dev0`.

7. If the versions were updated in step 3, update also the [conda forge recipe](https://github.com/conda-forge/arviz-feedstock).

   :::{important}
   There is a bot that opens a PR automatically to update the conda forge recipe.
   There is a second bot that also attempts to update dependencies automatically from `pyproject.toml`,
   however, at the time of writing it is experimental, so we always need to double check dependencies.
   :::

8. (here for reference but we really want to avoid it) If for some reason there were an issue
   with a release, there are two tools to fix it in conda forge:

   * repodata patch: If there is a new release for one of the dependencies with breaking changes
     for example, a repodata patch should be submitted to conda forge to prevent users
     from installing a broken environment.

     Repodata patches are submitted to [conda-forge/conda-forge-repodata-patches-feedstock](https://github.com/conda-forge/conda-forge-repodata-patches-feedstock)
   * mark a build as broken: If the dependencies were incorrect in the recipe, then the existing
     build should be marked as broken and a PR (completely manual) should be submitted
     to the arviz-feedstock to fix the recipe. Note that if this is being done for a release
     different than the latest, changes should not be merged into `main` but on a dedicated
     branch, the conda-forge package build is generated all the same but the history is kept
     tidier and prevents issues when a new release is published.

     Requests to mark packages as broken are submitted to [conda-forge/admin-requests](https://github.com/conda-forge/admin-requests/)
