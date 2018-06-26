#!/usr/bin/env bash

set -e # fail on first error

if conda --version > /dev/null 2>&1; then
   echo "conda appears to already be installed"
   exit 0
 fi

INSTALL_FOLDER="$HOME/miniconda"


if [ ! -d $INSTALL_FOLDER ] || [ ! -e $INSTALL_FOLDER/bin/conda ]; then
  if [ "$(uname)" == "Darwin" ]; then
    URL_OS="MacOSX"
  elif [ "$(expr substr "$(uname -s)" 1 5)" == "Linux" ]; then
    URL_OS="Linux"
  elif [ "$(expr substr "$(uname -s)" 1 10)" == "MINGW32_NT" ]; then
    URL_OS="Windows"
  fi

  echo "Downloading miniconda for $URL_OS"
  DOWNLOAD_PATH="miniconda.sh"
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-$URL_OS-x86_64.sh -O ${DOWNLOAD_PATH};

  echo "Installing miniconda to $INSTALL_FOLDER"
  # install miniconda to home folder
  bash ${DOWNLOAD_PATH} -b -f -p $INSTALL_FOLDER

  # tidy up
  rm ${DOWNLOAD_PATH}
else
  echo "Miniconda already installed at ${INSTALL_FOLDER}.  Updating, adding to path and exiting"
fi

export PATH="$INSTALL_FOLDER/bin:$PATH"
echo "Adding $INSTALL_FOLDER to PATH.  Consider adding it in your .rc file as well."
conda update -q -y conda
