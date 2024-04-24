from __future__ import annotations

import setuptools
from deepcave import version


def read_file(file_name):
    with open(file_name, encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        # Tests
        "pytest>=4.6",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.3.1",
        "pytest-timeout>=2.2.0",
        "mypy>=1.6.1",
        "isort>=5.12.0",
        "black>=23.11.0",
        "pydocstyle>=6.3.0",
        "pre-commit>=3.5.0",
        "flake8>=6.1.0",
        # Docs
        "automl-sphinx-theme>=0.1.10",
    ],
    "examples": [
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "pytorch-lightning>=2.1.1",
    ],
}


setuptools.setup(
    name="deepcave",
    author_email="s.segel@ai.uni-hannover.de",
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
    python_requires=">=3.9, <3.11",
    install_requires=read_file("./requirements.txt").split("\n"),
    extras_require=extras_require,
    entry_points={
        "console_scripts": ["deepcave = deepcave.cli:main"],
    },
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
