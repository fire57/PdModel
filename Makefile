PYTHON ?= .venv/bin/python
DVC := .venv/bin/dvc
API_URL ?= http://127.0.0.1:8000

.PHONY: install quality test validate prepare train evaluate repro api mlflow smoke monitor monitor-shift export

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

quality:
	$(PYTHON) -m black --check src tests scripts
	$(PYTHON) -m flake8 src tests scripts

test:
	$(PYTHON) -m pytest

validate:
	$(PYTHON) -m pd_model.data.validation --data data/raw/UCI_Credit_Card.csv --suite raw --report reports/validation/raw.json

prepare:
	PATH="$(CURDIR)/.venv/bin:$$PATH" $(DVC) repro prepare

train:
	PATH="$(CURDIR)/.venv/bin:$$PATH" $(DVC) repro train

evaluate:
	PATH="$(CURDIR)/.venv/bin:$$PATH" $(DVC) repro evaluate

repro:
	PATH="$(CURDIR)/.venv/bin:$$PATH" $(DVC) repro

api:
	$(PYTHON) -m uvicorn pd_model.api.app:app --host 127.0.0.1 --port 8000

mlflow:
	$(PYTHON) -m mlflow server --backend-store-uri sqlite:///mlflow.db --no-serve-artifacts --host 127.0.0.1 --port 5000

smoke:
	$(PYTHON) scripts/smoke_api.py --url $(API_URL)

monitor:
	$(PYTHON) -m pd_model.monitoring.simulate --scenario normal --api-url $(API_URL)

monitor-shift:
	$(PYTHON) -m pd_model.monitoring.simulate --scenario shifted --api-url $(API_URL)

export:
	$(DVC) push
	$(PYTHON) scripts/export_dvc_remote.py
	$(PYTHON) scripts/archive_tracking.py export
