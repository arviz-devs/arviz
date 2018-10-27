FROM conda/miniconda3


MAINTAINER Ravin Kumar

ARG SRC_DIR

ARG PYTHON_VERSION
ARG PYSTAN_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV PYSTAN_VERSION=${PYSTAN_VERSION}

RUN echo ${PYTHON_VERSION}
RUN apt-get update && apt-get install -y git gcc g++ gfortran && rm -rf /var/lib/apt/lists/*

COPY  $SRC_DIR /opt/arviz

WORKDIR /opt/arviz
# Clear any cached models from copied from host filesystem
RUN rm arviz/tests/saved_models/*.pkl
RUN find -type d -name __pycache__ -exec rm -rf {} +

# Create conda environment. Defaults to Python 3.6
RUN  ./scripts/create_testenv.sh

RUN conda clean --all

CMD ["/bin/bash", "-c", "source /root/.bashrc && pytest -vv arviz/tests"]