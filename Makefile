PYTHON ?= .venv/bin/python

.PHONY: help test check showcase demo data task1 task2 task34 task5 task6 task56-check
help:
	@echo 'test: synthetic tests | check: public repository checks | showcase: export selected local outputs'
	@echo 'data / task1 / task2 / task34 / task5 / task6 require original inputs; see docs/REPRODUCIBILITY.md'

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	python3 scripts/export_showcase.py --check
	python3 scripts/check_repository.py

showcase:
	python3 scripts/export_showcase.py
	$(PYTHON) scripts/build_portfolio.py

demo:
	python3 -m http.server 8000

data:
	$(PYTHON) scripts/prepare_data.py
	$(PYTHON) scripts/plot_examples.py
	$(PYTHON) scripts/prepare_dataset.py

task1:
	OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $(PYTHON) scripts/train_task1.py --threads 4
	$(PYTHON) scripts/report_task1.py
	$(PYTHON) scripts/check_task1.py

task2:
	OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $(PYTHON) scripts/discover_task2.py
	$(PYTHON) scripts/report_task2.py

task34:
	$(PYTHON) scripts/download_conditions.py
	$(PYTHON) scripts/prepare_conditions.py
	OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $(PYTHON) scripts/analyze_conditions.py
	$(PYTHON) scripts/check_conditions.py
	$(PYTHON) scripts/report_conditions.py

task5:
	OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $(PYTHON) scripts/train_task5.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(PYTHON) scripts/diagnose_task5.py

task6:
	$(PYTHON) scripts/prepare_task6.py
	OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $(PYTHON) scripts/train_task6.py

task56-check:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(PYTHON) scripts/check_task56.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(PYTHON) scripts/report_task56.py
