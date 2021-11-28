# Library architecture
ArviZ is organized in modules (the folders in [arviz directory](https://github.com/arviz-devs/arviz/tree/main/arviz)).
The main 3 modules are `data`, `plots` and `stats`.
Then we have 3 more folders. The [tests](https://github.com/arviz-devs/arviz/tree/main/arviz/tests)
folder contains tests for all these 3 modules.

The [static](https://github.com/arviz-devs/arviz/tree/main/arviz/static)
folder is only used to store style and CSS files to get HTML output for `InferenceData`.
Finally we have the [wrappers](https://github.com/arviz-devs/arviz/tree/main/arviz/wrappers)
folder that contains experimental (not tested yet either) features
and interacts closely with both [data](https://github.com/arviz-devs/arviz/tree/main/arviz/data)
and [stats](https://github.com/arviz-devs/arviz/tree/main/arviz/stats) modules.

In addition, there are some files on the higher level directory: `utils.py`, `sel_utils.py`,
`rcparams.py` and `labels.py`.
