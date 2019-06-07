import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_zoo",
    version="1.0.0",
    author="Bilal Khan",
    author_email="bk@tinymanager.com",
    description="A collection of useful modules and utilities for kaggle not available in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bkahn-github/pytorch_zoo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
