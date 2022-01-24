# -*- encoding: utf-8 -*-
import sys
import setuptools


def read_file(file_name):
    with open(file_name, encoding="utf-8") as fh:
        text = fh.read()
    return text


extras_require = {
    "formatting": ["flake8", "black", "isort"],
    "tests": [
        "pytest>=4.6",
        "mypy",
        "pytest-xdist",
        "pytest-timeout",
        "openml",
        "pre-commit",
        "pytest-cov",
    ],
    "examples": [
        "matplotlib",
        "jupyter",
        "notebook",
        "seaborn",
    ],
    "docs": [
        "sphinx<4.3",
        "sphinx-gallery",
        "sphinx_bootstrap_theme",
        "numpydoc",
        "sphinx_toolbox",
        "docutils==0.16",
    ],
}


setuptools.setup(
    name="deepcave",
    author_email="{sass, lindauer}@tnt.uni-hannover.de",
    description="An interactive framework to visualize and analyze your AutoML process in real-time.",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://www.automl.org",
    project_urls={
        "Documentation": "https://github.com/automl/deepcave",
        "Source Code": "https://github.com/automl/deepcave",
    },
    version=read_file("deepcave/__version__.py").split()[-1].strip("'"),
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    python_requires=">3.8, <=3.10",
    install_requires=read_file("./requirements.txt").split("\n"),
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
