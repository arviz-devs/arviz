from setuptools import setup, find_packages
from setuptools.command.install import install


class InstallStyles(install):
    def run(self):
        import sys
        import shutil
        import os
        from matplotlib import get_configdir

        sd = os.path.dirname(os.path.realpath(__file__))
        lsd = os.path.join(sd, 'arviz', 'plots', 'styles')
        styles = [f for f in os.listdir(lsd)]
        if not os.path.isdir(sd):
            os.makedirs(sd)
        for s in styles:
            shutil.copy(os.path.join(lsd, s), os.path.join(sd, s))
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
    cmdclass={'install': InstallStyles},
)
