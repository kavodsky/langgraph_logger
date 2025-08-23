# Makefile
.PHONY: install test lint format clean build publish

install:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/langgraph_logger --cov-report=html --cov-report=term

lint:
	mypy src/langgraph_logger
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish-test: build
	python -m twine upload --repository testpypi dist/*

publish: build
	python -m twine upload dist/*