# Usage:  make <target> [GRID=quick|full] [MOCK=1]
# Stub scripts print "not implemented yet" and exit 0, so `make all` never breaks the chain.

PYTHON ?= python
GRID   ?= quick
MOCK   ?=
ARGS   := --grid $(GRID) $(if $(MOCK),--mock,)

.PHONY: install test lint data functional cache costs results demo all

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

data:        # P1: shots + graphs
	$(PYTHON) scripts/p1_make_data.py $(ARGS)

functional:  # P1: functional traces
	$(PYTHON) scripts/p1_run_functional.py $(ARGS)

cache:       # P2: cache traces
	$(PYTHON) scripts/p2_build_cache.py $(ARGS)

costs:       # P2: service tables + iter_costs.json
	$(PYTHON) scripts/p2_build_costs.py $(ARGS)

results:     # P3: figures + tables
	$(PYTHON) scripts/p3_make_results.py $(ARGS)

demo:        # P3: live demo
	$(PYTHON) scripts/p3_demo.py $(ARGS)

all: data functional cache costs results
