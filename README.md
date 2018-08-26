<img src="doc/_static/logo.png" height=100></img>

[![Build Status](https://travis-ci.org/arviz-devs/arviz.svg?branch=master)](https://travis-ci.org/arviz-devs/arviz) [![Coverage Status](https://coveralls.io/repos/github/arviz-devs/arviz/badge.svg?branch=master)](https://coveralls.io/github/arviz-devs/arviz?branch=master)

# ArviZ
ArviZ (pronounced "AR-_vees_") is a Python package for exploratory analysis of Bayesian models.
Includes functions for posterior analysis, model checking, comparison and diagnostics.

## Documentation
The official Arviz documentation can be found here  
http://arviz-devs.github.io/arviz/index.html

## Installation

The latest version can be installed from the master branch using pip:

```
pip install git+git://github.com/arviz-devs/arviz.git
```

Another option is to clone the repository and install using `python setup.py install`.

## Dependencies

Arviz is tested on Python 3.6 and depends on NumPy, SciPy, Pandas and Matplotlib.

## Developing

There is a Dockerfile which helps for isolating build problems and local development. Install Docker for your operating system, clone this repo, then run ./scripts/start_container.sh. This should start a local docker container called arviz, as well as a Jupyter notebook server running on port 8888. The notebook should be opened in your browser automatically (you can disable this by passing --no-browser). The container will be running the code from your local copy of arviz, so you can test your changes.
