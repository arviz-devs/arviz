#! /bin/bash
SRC_DIR=${SRC_DIR:-`pwd`}

# Build container for use of testing or notebook
if [[ $* == *--build* ]]; then
    echo "Building Docker Image"
    docker build \
        -t arviz \
        -f $SRC_DIR/scripts/Dockerfile \
        --build-arg SRC_DIR=. $SRC_DIR \
        --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
        --build-arg PYSTAN_VERSION=${PYSTAN_VERSION} \
        --build-arg PYRO_VERSION=${PYRO_VERSION} \
        --build-arg EMCEE_VERSION=${EMCEE_VERSION} \
        --rm
fi

if [[ $* == *--clear_cache* ]]; then
    echo "Removing cached files and models"
    find -type d -name __pycache__ -exec rm -rf {} +
    rm -f arviz/tests/saved_models/*.pkl
fi

if [[ $* == *--test* ]]; then
    echo "Testing Arviz"
    docker run --mount type=bind,source="$(pwd)",target=/opt/arviz/ arviz:latest bash -c \
                                      "NUMBA_DISABLE_JIT=1 pytest -v arviz/tests/ --cov=arviz/"
fi
