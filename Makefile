.PHONY: install run debug clean lint lint-strict test
.PHONY: index search search_dataset answer answer_dataset evaluate
.PHONY: recall recall-docs recall-code subject-check moulinette

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
#   make answer_dataset ARGS="--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer"
#   make evaluate ARGS="--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/evaluate_results/UnansweredQuestions"
ARGS ?=
DATASET_DOCS ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
DATASET_CODE ?= data/datasets/UnansweredQuestions/dataset_code_public.json
GROUND_TRUTH_DOCS ?= data/datasets/AnsweredQuestions/dataset_docs_public.json
GROUND_TRUTH_CODE ?= data/datasets/AnsweredQuestions/dataset_code_public.json
SEARCH_OUT_DOCS ?= data/output/search_results/UnansweredQuestions
SEARCH_OUT_CODE ?= data/output/search_results/UnansweredQuestions
MOUSETTE_BIN ?= $(firstword $(wildcard moulinette) $(wildcard moulinette-ubuntu) $(wildcard moulinette-fedora))

ifeq ($(DETECTED_OS),Windows)
install:
	uv pip install -e .
else
install:
	test -d $(UV_PROJECT_ENVIRONMENT) || uv venv $(UV_PROJECT_ENVIRONMENT)
	VIRTUAL_ENV=$(UV_PROJECT_ENVIRONMENT) uv pip install -e .
endif

moulinette:
	@if [ -x moulinette-ubuntu ]; then ln -sf moulinette-ubuntu moulinette; fi
	@if [ ! -x moulinette ]; then \
		echo "moulinette binary not found. Rename moulinette-ubuntu to moulinette first."; \
		exit 1; \
	fi

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

# Official subject evaluation flow: index -> search_dataset -> moulinette evaluate_student_search_results
recall-docs: moulinette
	uv run python -m src index --max_chunk_size 2000
	uv run python -m src search_dataset --dataset_path $(DATASET_DOCS) --k 10 --save_directory $(SEARCH_OUT_DOCS)
	./moulinette evaluate_student_search_results $(SEARCH_OUT_DOCS)/dataset_docs_public.json $(GROUND_TRUTH_DOCS) --k 10 --max_context_length 2000

recall-code: moulinette
	uv run python -m src index --max_chunk_size 2000
	uv run python -m src search_dataset --dataset_path $(DATASET_CODE) --k 10 --save_directory $(SEARCH_OUT_CODE)
	./moulinette evaluate_student_search_results $(SEARCH_OUT_CODE)/dataset_code_public.json $(GROUND_TRUTH_CODE) --k 10 --max_context_length 2000

recall: recall-docs

subject-check: recall-docs recall-code

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