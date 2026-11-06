.PHONY: install run debug clean lint lint-strict test help check validate
.PHONY: index search search_dataset answer answer_dataset evaluate
.PHONY: recall recall-docs recall-code subject-check moulinette

# Environment variables for caching and virtual environment.
ifdef CACHE_DIR
export UV_CACHE_DIR=$(CACHE_DIR)/uv_cache
export UV_PROJECT_ENVIRONMENT=$(CACHE_DIR)/venv
export HF_HOME=$(CACHE_DIR)/hf_cache
else
export UV_CACHE_DIR=/goinfre/$(USER)/RAG_cache
export UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/RAG_venv
export HF_HOME=/goinfre/$(USER)/hf_cache
endif
unexport VIRTUAL_ENV

# Default path for moulinette files
ARGS ?=
DATASET_DOCS ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
DATASET_CODE ?= data/datasets/UnansweredQuestions/dataset_code_public.json
GROUND_TRUTH_DOCS ?= data/datasets/AnsweredQuestions/dataset_docs_public.json
GROUND_TRUTH_CODE ?= data/datasets/AnsweredQuestions/dataset_code_public.json
SEARCH_OUT_DOCS ?= data/output/search_results/UnansweredQuestions
SEARCH_OUT_CODE ?= data/output/search_results/UnansweredQuestions
SEARCH_RESULT_DOCS_FILE ?= $(SEARCH_OUT_DOCS)/$(notdir $(DATASET_DOCS))
SEARCH_RESULT_CODE_FILE ?= $(SEARCH_OUT_CODE)/$(notdir $(DATASET_CODE))
MOULI_BIN ?= $(firstword $(wildcard moulinette-ubuntu))


#   make help
#   make install
#   make index ARGS="--max_chunk_size 2000"
#   make search ARGS="'how to load a LoRA adapter' --k 10"
#   make search_dataset ARGS="--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions"
#   make answer ARGS="'how to load a LoRA adapter' --k 10"
#   make answer_dataset ARGS="--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer"
#   make evaluate ARGS="--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/evaluate_results/UnansweredQuestions"
#   make recall-docs
#   make recall-code
#   make subject-check
#   make check

#	docker run -it --rm -v .:/app -w /app -v ragcache:/cache -e CACHE_DIR=/cache --gpus all ghcr.io/astral-sh/uv:debian make search ARGS="'how to load a LoRA adapter' --k 10"


install:
	@echo "Syncing..."
	@uv sync

moulinette:
ifeq ($(MOULI_BIN), moulinette-ubuntu)
	@cp -f moulinette-ubuntu moulinette
	@chmod +x moulinette
else ifeq ($(MOULI_BIN), moulinette)
	@chmod +x moulinette
else
	@echo "moulinette binary not found. Rename moulinette-ubuntu to moulinette first."
	@exit 1
endif

run:
	@uv run python -m src $(ARGS)

# RAG commands
index:
	@uv run python -m src index $(ARGS)

search:
	@uv run python -m src search $(ARGS)

search_dataset:
	@uv run python -m src search_dataset $(ARGS)

answer:
	@uv run python -m src answer $(ARGS)

answer_dataset:
	@uv run python -m src answer_dataset $(ARGS)

evaluate:
	@uv run python -m src evaluate $(ARGS)


# subject evaluation commands
recall-docs: moulinette
	@uv run python -m src index --max_chunk_size 2000
	@uv run python -m src search_dataset --dataset_path $(DATASET_DOCS) --k 10 --save_directory $(SEARCH_OUT_DOCS)
	@./moulinette evaluate_student_search_results $(SEARCH_RESULT_DOCS_FILE) $(GROUND_TRUTH_DOCS) --k 10 --max_context_length 2000

recall-code: moulinette
	@uv run python -m src index --max_chunk_size 2000
	@uv run python -m src search_dataset --dataset_path $(DATASET_CODE) --k 10 --save_directory $(SEARCH_OUT_CODE)
	@./moulinette evaluate_student_search_results $(SEARCH_RESULT_CODE_FILE) $(GROUND_TRUTH_CODE) --k 10 --max_context_length 2000

recall: recall-docs recall-code

validate: lint-strict test
	@echo "Running recall validation..."
	$(MAKE) recall

# misc commands
help:
	@echo "Common commands:"
	@echo "  make install"
	@echo "  make index ARGS='--max_chunk_size 2000'"
	@echo "  make search ARGS='\"how to load a LoRA adapter\" --k 10'"
	@echo "  make search_dataset ARGS='--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions'"
	@echo "  make recall-docs"
	@echo "  make recall-code"
	@echo "  make validate"
	@echo "  make lint"

debug:
	@uv run python -m pdb src/main.py $(ARGS)

clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ .mypy_cache
	@rm -rf src/__pycache__ src/*.egg-info
	@rm -rf data/processed

lint:
	@echo "Running flake8..."
	@-uv run flake8 src --exclude .venv
	@echo "Running mypy..."
	@-uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: lint
	@echo "Running mypy strict..."
	-uv run mypy src --strict

test:
	@echo "Running pytest..."
	@uv run pytest tests --maxfail=1 --disable-warnings -q