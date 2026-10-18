PYTHON ?= conda run --no-capture-output -n microcture python -B

.PHONY: help check test demo serve
help:
	$(PYTHON) run.py --help
check:
	$(PYTHON) run.py check
test:
	$(PYTHON) run.py check
demo:
	$(PYTHON) run.py demo
serve:
	$(PYTHON) run.py serve
