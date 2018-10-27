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
        --build-arg PYSTAN_VERSION=${PYSTAN_VERSION}\
        --rm \
        --no-cache
fi
