.PHONY: install run debug clean lint lint-strict test
.PHONY: index search search_dataset answer answer_dataset evaluate

# uses /goinfre to save space
export UV_CACHE_DIR     = /goinfre/$(USER)/RAG_cache
export UV_PROJECT_ENVIRONMENT = /goinfre/$(USER)/RAG_venv
export HF_HOME          = /goinfre/$(USER)/hf_cache
unexport VIRTUAL_ENV

# Arguments for run
ARGS ?=

install:
	test -d $(UV_PROJECT_ENVIRONMENT) || uv venv $(UV_PROJECT_ENVIRONMENT)
	VIRTUAL_ENV=$(UV_PROJECT_ENVIRONMENT) uv pip install -e .

run:
	uv run python -m src $(ARGS)

# Aliases for each subcommand
index:
	$(MAKE) run ARGS="index $(ARGS)"

search:
	$(MAKE) run ARGS="search $(ARGS)"

search_dataset:
	$(MAKE) run ARGS="search_dataset $(ARGS)"

answer:
	$(MAKE) run ARGS="answer $(ARGS)"

answer_dataset:
	$(MAKE) run ARGS="answer_dataset $(ARGS)"

evaluate:
	$(MAKE) run ARGS="evaluate $(ARGS)"

# Other targets
debug:
	uv run python -m pdb src/main.py $(ARGS)

clean:
	rm -rf __pycache__ .mypy_cache

lint:
	uv run flake8 src --exclude .venv
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: lint
	uv run mypy src --strict

test:
	uv run pytest tests --maxfail=1 --disable-warnings -q