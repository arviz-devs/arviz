# Updating the example data

Metadata for ArviZ's example `InferenceData` objects (which are loadable by {func}`~arviz.load_arviz_data`) is stored in the [`arviz_example_data`](https://github.com/arviz-devs/arviz_example_data) repository.
This repo has been embedded into the `arviz` repo at `/arviz/data/example_data` using [git subtree](https://www.atlassian.com/git/tutorials/git-subtree) with the following command:

```bash
$ git subtree add --prefix arviz/data/example_data https://github.com/arviz-devs/arviz_example_data.git main --squash
```

When `arviz_example_data` is updated, the subtree within the `arviz` repo also needs to be updated with the following command:

```bash
$ git subtree pull --prefix arviz/data/example_data https://github.com/arviz-devs/arviz_example_data.git main --squash
```
