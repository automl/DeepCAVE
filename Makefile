# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

SHELL := /bin/bash
VERSION := 1.3.4

NAME := DeepCAVE
PACKAGE_NAME := deepcave

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DIST := dist
DOCDIR := docs
INDEX_HTML := "file://${DIR}/docs/build/html/index.html"
EXAMPLES_DIR := examples
TESTS_DIR := tests

.PHONY: help install install-dev check format pre-commit clean docs clean-doc examples clean-build build publish test

help:
	@echo "Makefile ${NAME}"
	@echo "* install          to install DeepCAVE locally"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* build            to build a dist"
	@echo "* docs             to generate and view the html files, checks links"
	@echo "* examples         to run and generate the examples"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python
PYTEST ?= python -m pytest
PIP ?= pip
MAKE ?= make
BLACK ?= black
ISORT ?= isort
PYDOCSTYLE ?= pydocstyle
MYPY ?= mypy
PRECOMMIT ?= pre-commit
FLAKE8 ?= flake8

install:
	conda install -y numpy=2.0.1 
	$(PIP) install -e .

# Fix numpy as version 2.1.0 will drop support for Python 3.9
install-dev:
	conda install -y numpy=2.0.1 
	$(PIP) install -e ".[dev]"
	pre-commit install

install-examples:
	$(PIP) install -e ".[examples]"

install-optuna:
	$(PIP) install -e ".[optuna]"

install-bohb:
	$(PIP) install -e ".[bohb]"

check-black:
	$(BLACK) ${SOURCE_DIR} --check || :
	$(BLACK) ${EXAMPLES_DIR} --check || :
	$(BLACK) ${TESTS_DIR} --check || :

check-isort:
	$(ISORT) ${SOURCE_DIR} --check || :
	$(ISORT) ${TESTS_DIR} --check || :

check-pydocstyle:
	$(PYDOCSTYLE) ${SOURCE_DIR} || :

check-mypy:
	$(MYPY) --check-untyped-defs --install-types --non-interactive --ignore-missing-imports ${SOURCE_DIR} || :

check-flake8:
	$(FLAKE8) ${SOURCE_DIR} || :
	$(FLAKE8) ${TESTS_DIR} || :

check: check-black check-isort check-mypy check-flake8 check-pydocstyle

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) ${SOURCE_DIR}
	$(BLACK) ${TESTS_DIR}
	$(BLACK) ${EXAMPLES_DIR}

format-isort:
	$(ISORT) ${SOURCE_DIR}
	$(ISORT) ${TESTS_DIR}

format: format-black format-isort

test:
	$(PYTEST) ${TESTS_DIR}

clean-doc:
	$(MAKE) -C ${DOCDIR} clean

docs:
	$(MAKE) -C ${DOCDIR} docs
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

examples:
	$(MAKE) -C ${DOCDIR} examples
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py sdist

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean build
	read -p "Did you update the version number in Makefile and deepcave/__init__.py?"

	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uploaded distribution into"
	@echo "* Run the following:"
	@echo "--- pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${PACKAGE_NAME}==${VERSION}"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo "--- python -c 'import ${PACKAGE_NAME}'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo "--- python -m twine upload dist/*"

# Clean up any builds in ./dist as well as doc, if present
clean: clean-build clean-doc
