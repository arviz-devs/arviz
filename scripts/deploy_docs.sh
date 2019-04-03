#!/usr/bin/env bash

set -x # fail on first error, don't use -e because the token is a secret.

if [ "${TRAVIS_BRANCH}" = "master" ] && [ "${TRAVIS_PULL_REQUEST}" = "false" ]
then
    ghp-import -pfnr https://"${GH_TOKEN}"@github.com/"${TRAVIS_REPO_SLUG}".git doc/build
fi
