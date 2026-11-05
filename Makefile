.PHONY: help ingest clean features backtest predict demo test lint format

PYTHON ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

ingest:  ## Download raw race results from Jolpica-F1 API
	$(PYTHON) -m src.ingestion.jolpica

clean:  ## Clean raw data and produce data/clean/results.parquet
	$(PYTHON) -m src.cleaning.clean_results

features:  ## Build feature table from cleaned data
	$(PYTHON) -m src.pipeline.build_dataset

backtest:  ## Run walk-forward backtest
	$(PYTHON) -m src.pipeline.run_backtest

predict:  ## Interactive prediction mode
	$(PYTHON) -m src.prediction.predict

demo:  ## Quick demo with synthetic data
	$(PYTHON) -m src.prediction.predict --demo

test:  ## Run all tests
	$(PYTHON) -m pytest tests/ -v

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check src/ tests/

format:  ## Format with black
	$(PYTHON) -m black src/ tests/
