# How to make a new release

ArviZ uses the following process to cut a new release of the library.

1. Bump the version number in `arviz/__init__.py`

   ```diff
   - __version = "0.12.0"
   + __version = "0.12.1"
   ```

2. Update the release notes in `CHANGELOG.md`. Remove any subsections within the released version
   without any items within them.

   ```diff
   - ## v0.x.x Unreleased
   + ## v0.12.1 (2022 May 12)
   ```

   Empty subheadings for the "unreleased" development version don't need to be included yet.

3. Check versions in `requirements.txt` and `setup.py` files. Arviz aims to follow the recommendations in [SPEC-0](https://scientific-python.org/specs/spec-0000/) from scientific python.

4. Open a Pull Request including these changes. Make sure all CI tests pass, adding commits if necessary. Even if CI is passing on main, there might be new releases of dependencies that break CI.

5. Add a release in the Github [release page](https://github.com/arviz-devs/arviz/releases) once the PR is merged.

6. After the release on Github, the CI system will complete the rest of the steps. Including making any wheels and uploading the new version to PyPI.

7. Add a follow-up PR changing the version string to include the dev flag.
   Make sure the version string is [PEP 440](https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions) compliant.
   For example, after releasing `v0.12.1` it should be set to `0.13.0.dev0`.

8. Use the following template to add empty subheadings to the `CHANGELOG.md` file in the follow-up PR.

   ```markdown
   ## v0.x.x Unreleased

   ### New features

   ### Maintenance and fixes

   ### Deprecation

   ### Documentation
   ```

9. If the versions were updated in step 3, update also the [conda forge recipe](https://github.com/conda-forge/arviz-feedstock).
