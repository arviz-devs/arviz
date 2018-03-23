
from setuptools import setup

setup(
    name="arviz",
    version="0.1.0",
    packages=['arviz'],
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas'
    ],
)
