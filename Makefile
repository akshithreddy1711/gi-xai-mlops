PYTHON ?= python

.PHONY: install train app mlflow-ui clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.train --config configs/default.yaml

app:
	$(PYTHON) -c "import os, runpy; os.environ['APP_CONFIG_PATH']='configs/default.yaml'; runpy.run_path('app.py', run_name='__main__')"

mlflow-ui:
	mlflow ui --backend-store-uri $${MLFLOW_TRACKING_URI:-file:./mlruns}

clean:
	rm -rf __pycache__ .pytest_cache
