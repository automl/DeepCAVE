# NOTE: Used on linux, limited support outside of Linux
#
# A simple makefile to help with small tasks related to development of deepcave
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install-dev check format pre-commit clean clean-build build publish test

help:
	@echo "Makefile deepcave"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* clean            to clean the dist and doc build files"
	@echo "* build            to build a dist"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* tests            to run the tests"

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
ISORT ?= isort
PYDOCSTYLE ?= pydocstyle
MYPY ?= mypy
PRECOMMIT ?= pre-commit
FLAKE8 ?= flake8

DIR := ${CURDIR}
DIST := ${CURDIR}/dist

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

check-black:
	$(BLACK) deepcave examples tests --check || :

check-isort:
	$(ISORT) deepcave tests --check || :

check-pydocstyle:
	$(PYDOCSTYLE) deepcave || :

check-mypy:
	$(MYPY) deepcave || :

check-flake8:
	$(FLAKE8) deepcave || :
	$(FLAKE8) tests || :

# pydocstyle does not have easy ignore rules, instead, we include as they are covered
check: check-black check-isort check-mypy check-flake8 # check-pydocstyle

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) deepcave tests examples

format-isort:
	$(ISORT) deepcave tests

format: format-black format-isort

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Clean up any builds in ./dist as well as doc
clean:
	clean-build
	rm -rf cache

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py bdist

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean-build build
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following line:"
	@echo "pip install --index-url https://test.pypi.org/simple/ auto-sklearn"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo "python -m twine upload dist/*"

tests:
	$(PYTEST) tests


	