.PHONY: install run debug clean lint lint-strict test

export UV_CACHE_DIR=/goinfre/$(USER)/RAG_cache
export UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/RAG_venv
export HF_HOME=/goinfre/$(USER)/hf_cache
unexport VIRTUAL_ENV

ARG ?=

run:
	uv run python src/main.py $(ARG)

# Create virtual environment if it doesn't exist
install:
	test -d $(UV_PROJECT_ENVIRONMENT) || uv venv $(UV_PROJECT_ENVIRONMENT)
	VIRTUAL_ENV=$(UV_PROJECT_ENVIRONMENT) uv pip install -e .

debug:
	uv run python -m pdb main.py

clean:
	rm -rf __pycache__
	rm -rf .mypy_cache

lint:
	uv run flake8 src --exclude .venv
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: lint
	uv run mypy src --strict

test:
	uv run pytest tests --maxfail=1 --disable-warnings -q