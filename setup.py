import codecs
import os
import re

import setuptools
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
REQUIREMENTS_OPTIONAL_FILE = os.path.join(PROJECT_ROOT, "requirements-optional.txt")
REQUIREMENTS_DEV_FILE = os.path.join(PROJECT_ROOT, "requirements-dev.txt")
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "arviz", "__init__.py")


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_requirements_dev():
    with codecs.open(REQUIREMENTS_DEV_FILE) as buff:
        return buff.read().splitlines()


def get_requirements_optional():
    with codecs.open(REQUIREMENTS_OPTIONAL_FILE) as buff:
        return buff.read().splitlines()


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_version():
    lines = open(VERSION_FILE, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (VERSION_FILE,))


setup(
    name="arviz",
    version=get_version(),
    description="Exploratory analysis of Bayesian models",
    author="ArviZ Developers",
    url="http://github.com/arviz-devs/arviz",
    packages=find_packages(),
    install_requires=get_requirements(),
    extras_require=dict(all=get_requirements_optional()),  # test=get_requirements_dev(),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
)
