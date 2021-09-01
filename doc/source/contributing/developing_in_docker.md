
(developing_in_docker)=
# Developing in Docker

We have provided a Dockerfile which helps for isolating build problems, and local development.
Install [Docker](https://www.docker.com/) for your operating system, clone [this](https://github.com/arviz-devs/arviz) repo. Docker will generate an environment with your local copy of `arviz` with all the packages in Dockerfile.

## container.sh & container.ps1

Predefined docker commands can be run with a `./scripts/container.sh` (on Linux and macOS)
and with `./scripts/container.ps1`. The scripts enable developers easily to call predefined docker commands.
Users can use one or multiple flags.

They are executed in the following order: clear-cache, build, test, docs, shell, notebook, lab

### For Linux and macOS
    $ ./scripts/container.sh --clear-cache
    $ ./scripts/container.sh --build

    $ ./scripts/container.sh --test
    $ ./scripts/container.sh --docs
    $ ./scripts/container.sh --shell
    $ ./scripts/container.sh --notebook
    $ ./scripts/container.sh --lab

### For Windows
    $ powershell.exe -File ./scripts/container.ps1 --clear-cache
    $ powershell.exe -File ./scripts/container.ps1 --build

    $ powershell.exe -File ./scripts/container.ps1 --test
    $ powershell.exe -File ./scripts/container.ps1 --docs
    $ powershell.exe -File ./scripts/container.ps1 --shell
    $ powershell.exe -File ./scripts/container.ps1 --notebook
    $ powershell.exe -File ./scripts/container.ps1 --lab

## Testing in Docker
Testing the code using docker consists of executing the same file(`./scripts/container.ps1`) 3 times (you may need root privileges to run it).
1. First run `./scripts/container.sh --clear-cache`.
2. Then run `./scripts/container.sh --build`. This starts a local docker image called `arviz`.
3. Finally, run the tests with `./scripts/container.sh --test`.

The `./scripts/container.sh --build` command needs to be executed only once to build the docker image. If you have already built the docker image, and want to test your new changes, then it is a 2 step process.
1. Run `./scripts/container.sh --clear-cache` because the cache generated locally is not valid for docker.
2. Run `./scripts/container.sh --test` to test with docker.

**NOTE**: If you run into errors due to `__pycache__` files (i.e. while testing in
docker after testing locally or installing with pip after testing with
docker), try running `./scripts/container.sh --clear-cache` before the errored
command.

## Using the Docker image interactively
Once the Docker image is built with `./scripts/container.sh --build`, interactive containers can also be run. Therefore, code can be edited and executed using the docker container, but modifying directly the working directory of the host machine.

### For Linux and macOS
To start a bash shell inside Docker, run:

    $ docker run --mount type=bind,source="$(pwd)",target=/opt/arviz/ -it arviz bash

Alternatively, to start a jupyter notebook, there are two steps, first run:

    $ docker run --mount type=bind,source="$(pwd)",target=/opt/arviz/ --name jupyter-dock -it -d -p 8888:8888 arviz
    $ docker exec jupyter-dock bash -c "pip install jupyter"
    $ docker exec -it jupyter-dock bash -c "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"

### For Windows
For Windows, use %CD% on cmd.exe and $pwd.Path on powershell.

    $ docker run --mount type=bind,source=%CD%,target=/opt/arviz/ -it arviz bash

For running a jupyter notebook, run:

    $ docker run --mount type=bind,source=%CD%,target=/opt/arviz/ --name jupyter-dock -it -d -p 8888:8888 arviz
    $ docker exec jupyter-dock bash -c "pip install jupyter"
    $ docker exec -it jupyter-dock bash -c "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"

This will output something similar to `http://(<docker container id> or <ip>):8888/?token=<token id>`, and can be accessed at `http://localhost:8888/?token=<token id>`.

## Building documentation with Docker
The documentation can be built with Docker by running `./scripts/container.sh
--docs`. The docker image contains by default all dependencies needed
for building the documentation. After having build the docs in the Docker
container, they can be checked at `doc/build` and viewed in the browser from `doc/build/index.html`.

To update any file, go to the specific file(`.md`, `.rst`, or `.ipynb`) and make your changes. In order to check the results, you will have to rebuild only the docs, using the same command `./scripts/container.sh --docs`. The resulting docs will be saved in the doc/build directory as HTML files. Open your corresponding file and check the results.
