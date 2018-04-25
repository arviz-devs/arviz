from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import sys
import shutil
import os
from matplotlib import get_configdir


def copy_styles():
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


setup(
    name='arviz',
    version='0.1.0',
    description='Exploratory analysis of Bayesian models',
    author='ArviZ Developers',
    url="http://github.com/arviz-devs/arviz",
    packages=find_packages(),
    install_requires=['matplotlib',
                      'numpy',
                      'scipy',
                      'pandas'],
    include_package_data=True,
    cmdclass={
        'develop': DevelopStyles,
        'install': InstallStyles,
    },
)
