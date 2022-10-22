# How to make a new release

ArviZ uses the following process to cut a new release of the library.

1. Bump the version number in `arviz/__init__.py`

   ```diff
   - __version = "0.12.0"
   + __version = "0.12.1"
   ```

2. Update the release notes in `CHANGELOG.md`

   ```diff
   - ## v0.x.x Unreleased
   + ## v0.12.1 (2022 May 12)
   ```

   Empty subheadings don't need to be included yet.

3. Open a Pull Request including these changes. Make sure all CI tests pass, adding commits if necessary. Even if CI is passing on main, there might be new releases of dependencies that break CI.

4. Add a release in the Github [release page](https://github.com/arviz-devs/arviz/releases) once the PR is merged.

5. After the release on Github, the CI system will complete the rest of the steps. Including making any wheels and uploading the new version to PyPI.

6. Add a follow-up PR changing the version string to include the dev flag. Make sure the version string is [PEP 440](https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions) compliant.

7. Use the following template to add empty subheadings to the `CHANGELOG.md` file.

   ```markdown
   ## v0.x.x Unreleased

   ### New features

   ### Maintenance and fixes

   ### Deprecation

   ### Documentation
   ```