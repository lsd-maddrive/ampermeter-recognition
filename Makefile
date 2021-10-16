#* Variables
SHELL := /usr/bin/env bash
PYTHON := python

#################################
# Poetry install/uninstall
#################################
#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) - --uninstall

# Prepare project when you first time load it
init-project:
	poetry install
	poetry run pre-commit install
	poetry run nbdime config-git --enable
