.PHONY: install run debug clean lint lint-strict test
.PHONY: index search search_dataset answer answer_dataset evaluate

# --- OS detection ---
ifeq ($(OS),Windows_NT)
  DETECTED_OS := Windows
else
  DETECTED_OS := $(shell uname -s)
endif

# On Linux/macOS, use /goinfre paths to save disk space.
# On Windows, skip these exports (no /goinfre mount).
ifneq ($(DETECTED_OS),Windows)
  export UV_CACHE_DIR     = /goinfre/$(USER)/RAG_cache
  export UV_PROJECT_ENVIRONMENT = /goinfre/$(USER)/RAG_venv
  export HF_HOME          = /goinfre/$(USER)/hf_cache
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
endif
unexport VIRTUAL_ENV

# Arguments for run
# Usage examples:
#   make index ARGS="--max_chunk_size 2000"
#   make search ARGS="'how to load a LoRA adapter' --k 10"
#   make search_dataset ARGS="--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions"
#   make answer ARGS="'how to load a LoRA adapter' --k 10"
#   make answer_dataset ARGS="--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/answer_results/UnansweredQuestions"
#   make evaluate ARGS="--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/evaluate_results/UnansweredQuestions"
ARGS ?=

ifeq ($(DETECTED_OS),Windows)
install:
	uv pip install -e .
else
install:
	test -d $(UV_PROJECT_ENVIRONMENT) || uv venv $(UV_PROJECT_ENVIRONMENT)
	VIRTUAL_ENV=$(UV_PROJECT_ENVIRONMENT) uv pip install -e .
endif

run:
	uv run python -m src $(ARGS)

# Aliases for each subcommand (direct invocation: no recursion, so
# command-line ARGS are never overridden by a nested make).
index:
	uv run python -m src index $(ARGS)

search:
	uv run python -m src search $(ARGS)

search_dataset:
	uv run python -m src search_dataset $(ARGS)

answer:
	uv run python -m src answer $(ARGS)

answer_dataset:
	uv run python -m src answer_dataset $(ARGS)

evaluate:
	uv run python -m src evaluate $(ARGS)

# Other targets
debug:
	uv run python -m pdb src/main.py $(ARGS)

ifeq ($(DETECTED_OS),Windows)
clean:
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .mypy_cache rmdir /s /q .mypy_cache
	if exist src\__pycache__ rmdir /s /q src\__pycache__
	for /d %%d in (src\*.egg-info) do rmdir /s /q "%%d"
	if exist data\processed rmdir /s /q data\processed
else
clean:
	rm -rf __pycache__ .mypy_cache
	rm -rf src/__pycache__ src/*.egg-info
	rm -rf data/processed
endif

lint:
	-uv run flake8 src --exclude .venv
	-uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: lint
	-uv run mypy src --strict

test:
	uv run pytest tests --maxfail=1 --disable-warnings -q