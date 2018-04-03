
from setuptools import setup, find_packages

setup(
    name="arviz",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas'
    ],
)
