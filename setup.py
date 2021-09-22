from distutils.core import setup
from setuptools import find_packages

setup(
    name="tinycam",
    packages=find_packages(),
    version="0.0.1",
    license="MIT",
    description="Simple python library for camera calibration.",
    author="Mihhail Samusev",
    url="https://github.com/mihsamusev",
    keywords=["opencv", "calibration"]
)