#!/usr/bin/env bash

set -ex # fail on first error, print commands

command -v conda >/dev/null 2>&1 || {
  echo "Requires conda but it is not installed.  Run install_miniconda.sh." >&2;
  exit 1;
}

# if no python specified, use Travis version, or else 3.6
PYTHON_VERSION=${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION:-3.6}}
PYSTAN_VERSION=${PYSTAN_VERSION:-latest}

if [[ $* != *--global* ]]; then
    ENVNAME="testenv${PYTHON_VERSION}_PYSTAN${PYSTAN_VERSION}"

    if conda env list | grep -q ${ENVNAME}
    then
        echo "Environment ${ENVNAME} already exists, keeping up to date"
    else
        conda create -n ${ENVNAME} --yes pip python=${PYTHON_VERSION}
    fi

    source activate ${ENVNAME}
fi

# Pyro install with pip is ~511MB. These binaries are ~91MB, somehow, and do not
# break the build. The first is the Python 3.5 wheel, the second is 3.6.
if [ "$PYTHON_VERSION" = "3.5" ]; then
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
else
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
fi

if [ "$PYSTAN_VERSION" = "latest" ]; then
    pip install pystan
else
    pip install pystan==${PYSTAN_VERSION}
fi

conda install --yes numpy cython scipy pandas matplotlib pytest pylint sphinx numpydoc ipython xarray netcdf4 mkl-service

pip install --upgrade pip

#  Install editable using the setup.py
pip install -e .
pip install -r requirements-dev.txt
