#! /bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-`pwd`}

echo "Checking documentation..."
python -m pydocstyle --convention=numpy ${SRC_DIR}/arviz/
echo "Success!"

echo "Checking code style with black..."
<<<<<<< HEAD
python -m black -l 100 --check ${SRC_DIR}/arviz/ ${SRC_DIR}/examples/
=======
python -m black -l 100 --check ${SRC_DIR}/arviz/ ${SRC_DIR}/examples/ ${SRC_DIR}/asv_benchmarks/
>>>>>>> update fast_kde_2d benchmark and changelog
echo "Success!"

echo "Checking code style with pylint..."
python -m pylint ${SRC_DIR}/arviz/
echo "Success!"
