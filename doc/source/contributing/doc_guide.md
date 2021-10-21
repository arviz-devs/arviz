(doc_guide)=

# Documentation Guide

ArviZ documentation is built using a Python documentation tool, [Sphinx](https://www.sphinx-doc.org/en/master/). Sphinx converts `rst`(restructured text) files into HTML websites. There are different extensions availabel for converting other types of files into HTML websites like markdown, jupyter notebooks, etc.

Arviz [docs](https://github.com/arviz-devs/arviz/tree/main/doc/source) consist of `.rst`, `.md` and `.ipynb` files. It uses `myst-parser` and `myst-nb` for `.md` and `.ipynb` files, respectively. [Myst-parser](https://myst-parser.readthedocs.io/en/latest/sphinx/intro.html) parses all `.md` files as MyST(Markedly Structured Text).
Apart from `/doc`, ArviZ documentation also consists of docstrings. Docstrings are used in the `.py` files to explain the functions parameters and return values.

ArviZ docs also uses sphinx extensions for style, layout, navbar and putting code in the documentation. We will explore all the things one by one. Let's start!

```{toctree}
:maxdepth: 2

doc_dev_summary
docstrings
reference_guide
```
