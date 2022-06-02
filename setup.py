# -*- encoding: utf-8 -*-
import setuptools
from deepcave import version


def read_file(file_name):
    with open(file_name, encoding="utf-8") as fh:
        text = fh.read()
    return text


extras_require = {
    "dev": [
        # Tests
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        "mypy",
        "isort",
        "black",
        "pydocstyle",
        "pre-commit",
        "flake8",
        # Docs
        "automl-sphinx-theme>=0.1.10",
    ],
    "examples": [
        "torch",
        "torchvision",
        "pytorch-lightning",
    ],
}


setuptools.setup(
    name="deepcave",
    author_email="sass@tnt.uni-hannover.de",
    description="An interactive framework to visualize and analyze your AutoML process in real-time.",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://www.automl.org",
    project_urls={
        "Documentation": "https://github.com/automl/deepcave",
        "Source Code": "https://github.com/automl/deepcave",
    },
    version=version,
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    include_package_data=True,
    python_requires=">=3.9, <3.10",
    install_requires=read_file("./requirements.txt").split("\n"),
    extras_require=extras_require,
    entry_points={
        "console_scripts": ["deepcave = deepcave.cli:main"],
    },
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.9",
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
