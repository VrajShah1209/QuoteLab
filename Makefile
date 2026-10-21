.PHONY: install test smoke reproduce demo

install:            ## editable install with test + plotting extras
	pip install -e ".[dev]"

test:               ## 100 unit, property and fuzz tests (~10 s)
	pytest

demo:               ## one simulated episode, three strategies
	python -m quotelab demo

smoke:              ## the full research pipeline on a few short episodes (~6 min)
	QUOTELAB_FAST=1 python experiments/run_all.py

reproduce:          ## every number and figure in the report (~1.5 h on 2 cores)
	python experiments/run_all.py
