.PHONY: install test lint smoke download-data train clean

install:
	cd pii-ner-v1 && pip install -e ".[dev]"

test:
	cd pii-ner-v1 && pytest --tb=short -q

lint:
	cd pii-ner-v1 && ruff check src/ tests/ scripts/

lint-fix:
	cd pii-ner-v1 && ruff check --fix src/ tests/ scripts/

smoke:
	cd pii-ner-v1 && python -m scripts.smoke_test

download-data:
	cd pii-ner-v1 && python -m scripts.download_data

train:
	cd pii-ner-v1 && python -m datafog_pii_ner.training.train --config configs/default.yaml

clean:
	rm -rf pii-ner-v1/outputs/
	rm -rf pii-ner-v1/.ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
