.PHONY: init install format check

init:
	python -m venv .venv
	.venv/bin/python -m pip install -e ".[dev]"

install:
	.venv/bin/python -m pip install -e ".[dev]"

format:
	.venv/bin/python -m black pipelines core

check:
	.venv/bin/python -m py_compile pipelines/*.py core/*.py
	.venv/bin/python -m black --check pipelines core
