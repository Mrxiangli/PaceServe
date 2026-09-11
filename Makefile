.PHONY: help lint lint/flake8 lint/black lint/isort format format/black format/autopep8 format/isort
.DEFAULT_GOAL := help

lint/flake8: ## check style with flake8
	flake8 paceserve

lint/black: ## check style with black
	black --check paceserve

lint/isort: ## check style with isort
	isort --check-only --profile black paceserve

lint: lint/black lint/isort ## check style

format/black: ## format code with black
	black paceserve

format/autopep8: ## format code with autopep8
	autopep8 --in-place --recursive paceserve/

format/isort: ## format code with isort
	isort --profile black paceserve

format: format/black format/isort  ## format code
