from setuptools import setup, find_packages
from os import path

REQUIRED_PACKAGES = ["tensorflow == 2.2.0"]

this_directory = path.abspath(path.dirname(__file__))

setup(
    name="ocular",
    version="1.0",
    license='MIT',
    packages=find_packages(),
)