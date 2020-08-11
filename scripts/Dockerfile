FROM conda/miniconda3

LABEL maintainer="Ravin Kumar"

ARG SRC_DIR
ARG PYTHON_VERSION
ARG PYSTAN_VERSION
ARG PYRO_VERSION
ARG PYTORCH_VERSION
ARG EMCEE_VERSION
ARG TF_VERSION
ARG PYMC3_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV PYSTAN_VERSION=${PYSTAN_VERSION}
ENV PYRO_VERSION=${PYRO_VERSION}
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ENV EMCEE_VERSION=${EMCEE_VERSION}
ENV TF_VERSION=${TF_VERSION}
ENV PYMC3_VERSION=${PYMC3_VERSION}

# Change behavior of create_test.sh script
ENV DOCKER_BUILD=true

# For Sphinx documentation builds
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Update container
RUN apt-get update && apt-get install -y git build-essential pandoc vim ffmpeg \
    dos2unix libgtk-3-dev libdbus-glib-1-dev libxt-dev jags && rm -rf /var/lib/apt/lists/*

# Copy needed scripts
COPY $SRC_DIR/requirements*.txt /opt/arviz/
COPY $SRC_DIR/scripts /opt/arviz/scripts
RUN find /opt/arviz/ -type f -print0 | xargs -0 dos2unix

# Clear any cached models from copied from host filesystem
RUN rm -f arviz/tests/saved_models/*.pkl
    RUN find -type d -name __pycache__ -exec rm -rf {} +

# Change workdir
WORKDIR /opt/arviz

# Create conda environment. Defaults to Python 3.8
RUN ./scripts/create_testenv.sh

# Set automatic conda activation in non interactive and shells
ENV BASH_ENV="/root/activate_conda.sh"
RUN echo ". /root/activate_conda.sh" > /root/.bashrc

# Remove conda cache
RUN conda clean --all

# Copy ArviZ
COPY $SRC_DIR /opt/arviz
RUN find /opt/arviz/ -type f -print0 | xargs -0 dos2unix

# Clear any cached models from copied from host filesystem
RUN rm -f arviz/tests/saved_models/*.pkl
RUN find -type d -name __pycache__ -exec rm -rf {} +

# Install editable using the setup.py
RUN /bin/bash -c "source /root/.bashrc && python -m pip install -e ."
