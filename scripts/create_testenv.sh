#!/usr/bin/env bash

set -ex # fail on first error, print commands

command -v conda >/dev/null 2>&1 || {
  echo "Requires conda but it is not installed.  Run install_miniconda.sh." >&2;
  exit 1;
}

ENVNAME="testenv"
PYTHON_VERSION=${PYTHON_VERSION:-3.6} # if no python specified, use 3.6

if conda env list | grep -q ${ENVNAME}
then
    echo "Environment ${ENVNAME} already exists, keeping up to date"
else
    conda create -n ${ENVNAME} --yes pip python=${PYTHON_VERSION}
    source activate ${ENVNAME}
fi
conda install --yes numpy scipy pandas matplotlib pytest pylint sphinx numpydoc ipython xarray mkl-service

pip install --upgrade pip

#  Install editable using the setup.py
pip install -e .
pip install -r requirements-dev.txt