PYTHON ?= python

.PHONY: all download build model test
all: download build model
download:
	$(PYTHON) src/download_data.py
build:
	$(PYTHON) src/build_dataset.py
model:
	$(PYTHON) src/model.py
test:
	pytest -q
