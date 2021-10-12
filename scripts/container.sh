#! /bin/bash
SRC_DIR=${SRC_DIR:-`pwd`}

if [[ $* == *--clear-cache* ]]; then
    echo "Removing cached files and models"
    find -type d -name __pycache__ -exec rm -rf {} +
    rm -f arviz/tests/saved_models/*.pkl
    rm -f arviz/tests/saved_models/*.pkl.gzip
fi

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
        --build-arg PYTORCH_VERSION=${PYTORCH_VERSION} \
        --build-arg EMCEE_VERSION=${EMCEE_VERSION} \
        --build-arg TF_VERSION=${TF_VERSION} \
        --build-arg PYMC3_VERSION=${PYMC3_VERSION}
fi


if [[ $* == *--test* ]]; then
    echo "Testing Arviz"
    docker run --mount type=bind,source="$(pwd)",target=/opt/arviz/ --rm arviz:latest bash -c \
                                      "NUMBA_DISABLE_JIT=1 pytest -v arviz/tests/ --cov=arviz/"
fi


if [[ $* == *--docs* ]]; then
    echo "Build docs with sphinx"
    docker run  --mount type=bind,source="$(pwd)",target=/opt/arviz --name arviz_sphinx --rm arviz:latest bash -c \
    "if [ -d ./doc/build ]; then python -msphinx -M clean doc/source doc/build; fi && sphinx-build doc/source doc/build -b html"
    echo "Successful build. Find html docs in doc/build directory"
fi


if [[ $* == *--shell* ]]; then
    echo "Starting Arviz Container Shell"
    docker run  -it --mount type=bind,source="$(pwd)",target=/opt/arviz --name arviz_shell --rm arviz:latest /bin/bash
fi


if [[ $* == *--notebook* ]]; then
    echo "Starting Arviz Container Jupyter server"
    docker run --mount type=bind,source="$(pwd)",target=/opt/arviz/ --name jupyter-dock -it -d -p 8888:8888 arviz
    docker exec -it jupyter-dock bash -c "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"
fi


if [[ $* == *--lab* ]]; then
    echo "Starting Arviz Container Jupyter server with Jupyter Lab interface"
    docker run --mount type=bind,source="$(pwd)",target=/opt/arviz/ --name jupyter-dock -it -d -p 8888:8888 arviz
    docker exec -it jupyter-dock bash -c "jupyter lab --ip 0.0.0.0 --no-browser --allow-root"
fi
