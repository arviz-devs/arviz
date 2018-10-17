import codecs
import shutil
import os
import re

import setuptools
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')
README_FILE = os.path.join(PROJECT_ROOT, 'README.md')
VERSION_FILE = os.path.join(PROJECT_ROOT, 'arviz', '__init__.py')


# Ensure matplotlib dependencies are available to copy
# styles over
setuptools.dist.Distribution().fetch_build_eggs(['matplotlib>=3.0'])

def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_long_description():
    with codecs.open(README_FILE, 'rt') as buff:
        return buff.read()


def copy_styles():
    from matplotlib import get_configdir
    sd = os.path.join(get_configdir(), "stylelib")
    source = os.path.dirname(os.path.realpath(__file__))
    lsd = os.path.join(source, 'arviz', 'plots', 'styles')
    styles = [f for f in os.listdir(lsd)]
    if not os.path.isdir(sd):
        os.makedirs(sd)
    for s in styles:
        shutil.copy(os.path.join(lsd, s), os.path.join(sd, s))


class DevelopStyles(develop):
    def run(self):
        copy_styles()
        develop.run(self)

class InstallStyles(install):
    def run(self):
        copy_styles()
        install.run(self)

def get_version():
    lines = open(VERSION_FILE, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version in %s.' % (VERSION_FILE,))

setup(
    name='arviz',
    version=get_version(),
    description='Exploratory analysis of Bayesian models',
    author='ArviZ Developers',
    url="http://github.com/arviz-devs/arviz",
    packages=find_packages(),
    install_requires=get_requirements(),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    cmdclass={
        'develop': DevelopStyles,
        'install': InstallStyles,
    },
)
