PYTHON ?= python
.PHONY: all download build model demo app test clean
all: download build model
download:
	$(PYTHON) src/download_data.py
build:
	$(PYTHON) src/build_dataset.py
model:
	$(PYTHON) src/model.py
demo:
	$(PYTHON) scripts/generate_demo_data.py
	$(PYTHON) src/build_dataset.py
	$(PYTHON) src/model.py
	$(PYTHON) src/similarities.py --player "Demo Receiver 001"
app:
	streamlit run app.py
test:
	pytest -q
clean:
	rm -rf data/processed/* outputs/*

