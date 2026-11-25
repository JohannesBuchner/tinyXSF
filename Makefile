.PHONY: clean clean-test clean-pyc clean-build docs servedocs help install release release-test dist 
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

PYTHON := python3

BROWSER := $(PYTHON) -c "$$BROWSER_PYSCRIPT"

help:
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-doc ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.pyx.py' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '*.so' -exec rm -f {} +
	find tinyxsf -name '*.c' -exec rm -f {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-doc:
	rm -rf docs/build

SOURCES := $(shell ls tinyxsf/*.py)

lint: ${SOURCES} ## check style
	flake8 ${SOURCES}
	pycodestyle ${SOURCES}
	pydocstyle ${SOURCES}

test: ## run tests quickly with the default Python
	PYTHONPATH=. pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	PYTHONPATH=. coverage run --source tinyxsf -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/tinyxsf.rst
	rm -f docs/modules.rst
	rm -f docs/API.rst
	python3 setup.py build_ext --inplace
	sphinx-apidoc -H API -o docs/ tinyxsf
	cd docs; python3 modoverview.py
	$(MAKE) -C docs clean
	$(MAKE) -C docs html O=-jauto
	$(BROWSER) docs/build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload --verbose dist/*.tar.gz

dist: clean ## builds source and wheel package
	$(PYTHON) -m build
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	$(PYTHON) -m pip install .
	# xspec_models_cxc is not on PyPI; build it from the git repository
	# (this needs the XSPEC model library from the heasoft package, with
	# HEADAS set, which the CI harness provides via `mamba run`). It is
	# built after `pip install .` so that it is compiled against the
	# numpy version that the environment ends up with.
	# The xspec-models-cxc-helpers package expects the CXC xspec-modelsonly
	# conda layout, while the heasoft package uses the HEASARC layout, so
	# add compatibility symlinks: the helpers look for the XSPEC model
	# database at $$HEADAS/../spectral/manager/model.dat (the heasoft
	# package places it inside $$HEADAS), and for the linker name
	# libwcs.so (the heasoft package only ships libwcs.<version>.so).
	if [ -n "$$HEADAS" ] && [ ! -e "$$HEADAS/../spectral" ]; then ln -sfn "$$HEADAS/spectral" "$$HEADAS/../spectral"; fi
	if [ -n "$$HEADAS" ] && [ ! -e "$$HEADAS/lib/libwcs.so" ]; then (cd "$$HEADAS/lib" && ln -sfn "$$(ls libwcs.*.so | head -1)" libwcs.so); fi
	# g++ writes the temporary assembly file for the large generated
	# module to $TMPDIR, which defaults to /tmp; that filesystem can be a
	# small tmpfs, so keep the build temporaries next to the checkout
	mkdir -p .pipbuildtmp
	TMPDIR="$$(pwd)/.pipbuildtmp" $(PYTHON) -m pip install "xspec-models-cxc @ git+https://github.com/JohannesBuchner/xspec-models-cxc"
	rm -rf .pipbuildtmp
