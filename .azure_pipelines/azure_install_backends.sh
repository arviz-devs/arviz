#!/usr/bin/env bash

set -ex # fail on first error, print commands

# Install ArviZ dependencies
python -m pip install --upgrade pip

if [ "$(pytorch.version)" = "latest" ]; then
    python -m pip --no-cache-dir install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html
else
    python -m pip --no-cache-dir install torch==${pytorch.version} torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html
fi

if [ "$(pystan.version)" = "latest" ]; then
    python -m pip --no-cache-dir install pystan
else
    if [ "$(pystan.version)" = "preview" ]; then
        # try to skip other pre-releases than pystan
        python -m pip --no-cache-dir install numpy uvloop marshmallow==3.0.0rc6 PyYAML
        python -m pip --no-cache-dir install --pre pystan
    else
        python -m pip --no-cache-dir install pystan==${pystan.version}
    fi
fi

if [ "$(pyro.version)" = "latest" ]; then
    python -m pip --no-cache-dir install pyro-ppl
else
    python -m pip --no-cache-dir install pyro-ppl==${pyro.version}
fi

if [ "$(emcee.version)" = "latest" ]; then
    python -m pip --no-cache-dir install emcee
else
    python -m pip --no-cache-dir install "emcee<3"
fi

if [ "$(tensorflow.version)" = "latest" ]; then
    python -m pip --no-cache-dir install tensorflow
else
    python -m pip --no-cache-dir install tensorflow==1.14 tensorflow_probability==0.7
fi

if [ "$(pymc3.version)" = "latest" ]; then
    python -m pip --no-cache-dir install git+https://github.com/pymc-devs/pymc3
else
    python -m pip --no-cache-dir install pymc3==${pymc3.version}
fi

python -m pip install --no-cache-dir -r requirements-external.txt
