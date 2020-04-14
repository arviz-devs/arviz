if (!(Test-Path env:SRC_DIR)) {
  $SRC_DIR=$pwd.Path
}

$build = $false
$clearCache = $false
$docs = $false
$lab = $false
$notebook = $false
$shell = $false
$test = $false

foreach ($arg in $args) {
  if ($arg -eq "--build") {
    $build = $true
  }
  elseif ($arg -eq "--clear-cache") {
    $clearCache = $true
  }
  elseif ($arg -eq "--docs") {
    $docs = $true
  }
  elseif ($arg -eq "--lab") {
    $lab = $true
  }
  elseif ($arg -eq "--notebook") {
    $notebook = $true
  }
  elseif ($arg -eq "--shell") {
    $shell = $true
  }
  elseif ($arg -eq "--test") {
    $test = $true
  }
}

if ($build) {
  echo "Building Docker Image"
  # Build container for use of testing or notebook
  docker build \
      -t arviz \
      -f $SRC_DIR/scripts/Dockerfile `
      --build-arg SRC_DIR=. $SRC_DIR `
      --build-arg PYTHON_VERSION=${PYTHON_VERSION} `
      --build-arg PYSTAN_VERSION=${PYSTAN_VERSION} `
      --build-arg PYRO_VERSION=${PYRO_VERSION} `
      --build-arg PYTORCH_VERSION=${PYTORCH_VERSION} `
      --build-arg EMCEE_VERSION=${EMCEE_VERSION} `
      --build-arg TF_VERSION=${TF_VERSION} `
      --build-arg PYMC3_VERSION=${PYMC3_VERSION}
}

if ($clearCache) {
    echo "Removing cached files and models"
    Get-ChildItem -Path $SRC_DIR -Recurse -Force -Include "__pycache__" | Remove-Item -Force -Recurse
    Get-ChildItem -Path ($SRC_DIR + "\arviz\tests\saved_models") -Recurse -Force -Include "*.pkl" | Remove-Item -Force -Recurse
}

if ($test) {
  echo "Testing Arviz"
  docker run --mount type=bind,source=$SRC_DIR,target=/opt/arviz/ --rm arviz:latest bash -c `
  "NUMBA_DISABLE_JIT=1 pytest -v arviz/tests/ --cov=arviz/"
}

if ($docs) {
  echo "Build docs with sphinx"
  docker run  --mount type=bind,source=$SRC_DIR,target=/opt/arviz --name arviz_sphinx --rm arviz:latest bash -c `
  "if [ -d ./doc/build ]; then python -msphinx -M clean doc doc/build; fi && sphinx-build doc doc/build -b html"
}

if ($shell) {
  echo "Starting Arviz Container Shell"
  docker run  -it --mount type=bind,source=$SRC_DIR,target=/opt/arviz --name arviz_shell --rm arviz:latest /bin/bash
}

if ($notebook) {
  echo "Starting Arviz Container Jupyter server"
  docker run --mount type=bind,source=$SRC_DIR,target=/opt/arviz/ --name jupyter-dock -it -d -p 8888:8888 arviz
  docker exec -it jupyter-dock bash -c "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"
}

if ($lab) {
  echo "Starting Arviz Container Jupyter server with Jupyter Lab interface"
  docker run --mount type=bind,source=$SRC_DIR,target=/opt/arviz/ --name jupyter-dock -it -d -p 8888:8888 arviz
  docker exec -it jupyter-dock bash -c "jupyter lab --ip 0.0.0.0 --no-browser --allow-root"
}
