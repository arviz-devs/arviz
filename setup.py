import shutil
import os
import re

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as buff:
    install_reqs = buff.read().splitlines()


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
    versionfile = os.path.join('arviz', '__init__.py')
    lines = open(versionfile, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))

setup(
    name='arviz',
    version=get_version(),
    description='Exploratory analysis of Bayesian models',
    author='ArviZ Developers',
    url="http://github.com/arviz-devs/arviz",
    packages=find_packages(),
    install_requires=install_reqs,
    include_package_data=True,
    cmdclass={
        'develop': DevelopStyles,
        'install': InstallStyles,
    },
)
