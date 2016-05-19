
from setuptools import setup, find_packages

setup(
    name="mcmcplotlib",
    version="0.1.0",
    packages=['mcmcplotlib'],
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'
    ],
)
