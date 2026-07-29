# RAG against the machine — Project Timeline & Action Plan

> *This comprehensive guide outlines every step required to build, evaluate, and polish the Retrieval-Augmented Generation (RAG) system for the vLLM codebase. It serves as both a roadmap and a reference for mandatory features, evaluation thresholds, and optional bonus enhancements.*

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Phase 0: Project Setup & Foundation](#-phase-0-project-setup--foundation-day-1-2)
3. [Phase 1: Indexing](#-phase-1-indexing-day-3-4)
4. [Phase 2: Retrieval](#-phase-2-retrieval-day-5-6)
5. [Phase 3: Answer Generation](#-phase-3-answer-generation-day-7-8)
6. [Phase 4: Evaluation & Self-Testing](#-phase-4-evaluation--self-testing-day-9)
7. [Phase 5: Documentation & Polish](#-phase-5-documentation--polish-day-10)
8. [Phase 6: Bonus Features (Optional)](#-phase-6-bonus-features-optional-1-point-each)
9. [📅 Project Timeline Summary](#-project-timeline-summary)
10. [🚀 Critical Success Factors](#-critical-success-factors)
11. [⚙️ Consolidated CLI Commands](#%EF%B8%8F-consolidated-cli-commands)

---

## 🧠 Project Overview

You are building a **Retrieval-Augmented Generation (RAG)** system that answers questions about the vLLM repository. The pipeline consists of three main stages:

| Stage | Description |
|-------|-------------|
| **Indexing** | Read the entire codebase, intelligently chunk files (Python & Markdown), and build a searchable index. |
| **Retrieval** | Given a user question, find the top‑k most relevant code or documentation snippets. |
| **Answer Generation** | Feed those snippets to a lightweight local model (`Qwen/Qwen3-0.6B`) to produce a grounded, coherent answer. |

### 🎯 Evaluation Targets

- **Indexing time** ≤ 5 minutes.
- **Retrieval time** for 200 questions ≤ 90 seconds.
- **Recall@5** for documentation questions ≥ 80%.
- **Recall@5** for code questions ≥ 50%.

---

## 🏗️ Phase 0: Project Setup & Foundation (Day 1‑2)

*Goal: Establish the project skeleton, tooling, and core data models.*

| # | Task | Details / Reference |
|---|------|----------------------|
| 0.1 | **Initialize project** | `uv init` at the repository root. |
| 0.2 | **Define dependencies** | Add to `pyproject.toml`: `pydantic`, `python-fire`, `tqdm`, `flake8`, `mypy`, `transformers`, `torch`, `numpy`, `scikit-learn` (for TF‑IDF) or `rank-bm25`. |
| 0.3 | **Source structure** | Create `src/` directory with an `__init__.py` file. |
| 0.4 | **Makefile** | Include rules: `install`, `run`, `debug`, `clean`, `lint`. |
| 0.5 | **.gitignore** | Exclude `__pycache__/`, `.mypy_cache/`, `data/processed/`, `data/output/`. |
| 0.6 | **Pydantic models** | Implement in `src/models.py` the following classes: <br> • `MinimalSource` (file_path, first_character_index, last_character_index) <br> • `UnansweredQuestion` (question_id, question) <br> • `AnsweredQuestion` (inherits + sources, answer) <br> • `RagDataset` (rag_questions: List[AnsweredQuestion \| UnansweredQuestion]) <br> • `MinimalSearchResults` (question_id, question, retrieved_sources) <br> • `MinimalAnswer` (inherits + answer) <br> • `StudentSearchResults` (search_results, k) <br> • `StudentSearchResultsAndAnswer` (search_results, k) |
| 0.7 | **CLI entry point** | Set up `src/__main__.py` using Python Fire. |

---

## 📂 Phase 1: Indexing (Day 3‑4)

*Goal: Ingest the vLLM repository, chunk files intelligently, and build a lexical index.*

| # | Task | Description |
|---|------|-------------|
| 1.1 | **File discovery** | Recursively traverse `data/raw/` and collect all files. |
| 1.2 | **Python chunker** | Smart chunking for `.py` files — respect functions and classes. |
| 1.3 | **Markdown/text chunker** | Smart chunking for `.md` and text files — respect headings and paragraphs. |
| 1.4 | **Chunk size control** | Accept `--max_chunk_size` CLI argument (default: 2000 characters). |
| 1.5 | **Chunk metadata** | Each chunk stores: `file_path`, `start_char`, `end_char`, `content`. |
| 1.6 | **TF‑IDF index** | Build a TF‑IDF vectorizer over all chunks and persist it. |
| 1.7 | **BM25 alternative** | Implement BM25 as your lexical method (recommended for better baseline). |
| 1.8 | **Index persistence** | Save the index to `data/processed/`. |
| 1.9 | **Progress bar** | Add `tqdm` for long‑running operations. |
| 1.10 | **CLI command** | `uv run python -m src index --max_chunk_size 2000` |

> 💡 **Tip:** Start with BM25 — it often outperforms plain TF‑IDF on codebases and is straightforward to implement using `rank-bm25`.

---

## 🔍 Phase 2: Retrieval (Day 5‑6)

*Goal: Implement search over the index for single queries and batched datasets.*

| # | Task | Description |
|---|------|-------------|
| 2.1 | **Single query search** | Return top‑k chunks with `file_path` and character range. |
| 2.2 | **Ranking** | Score chunks by TF‑IDF/BM25 similarity to the query. |
| 2.3 | **Source validation** | Ensure each result’s `file_path` matches the corpus path exactly. |
| 2.4 | **CLI: `search`** | `uv run python -m src search "<query>" --k 10` |
| 2.5 | **Dataset search** | Read questions from JSON, run search for each, write `StudentSearchResults`. |
| 2.6 | **CLI: `search_dataset`** | `--dataset_path`, `--k`, `--save_directory` |
| 2.7 | **Output validation** | Ensure the output conforms to the `StudentSearchResults` model. |
| 2.8 | **Edge cases** | Gracefully handle empty queries, `k=0`, missing files, or malformed JSON. |
| 2.9 | **Progress bar** | Add `tqdm` for batch processing. |

---

## 🤖 Phase 3: Answer Generation (Day 7‑8)

*Goal: Feed retrieved snippets to the model and generate grounded, non‑hallucinated answers.*

| # | Task | Description |
|---|------|-------------|
| 3.1 | **Load Qwen/Qwen3‑0.6B** | Use `transformers` to load the model (CPU‑only setup). |
| 3.2 | **Prompt engineering** | Design a prompt that includes the retrieved context and asks the question clearly. |
| 3.3 | **Context window management** | Ensure all retrieved sources fit within the model’s token budget. |
| 3.4 | **Single answer** | Given a question and search results, generate an answer. |
| 3.5 | **CLI: `answer`** | `uv run python -m src answer "<query>" --k 10` |
| 3.6 | **Dataset answer generation** | Read `StudentSearchResults`, generate answers, write `StudentSearchResultsAndAnswer`. |
| 3.7 | **CLI: `answer_dataset`** | `--student_search_results_path`, `--save_directory` |
| 3.8 | **Groundedness** | Answers must be coherent and directly based on retrieved sources — no hallucinations. |

> 💡 **Insight from Call‑Me‑Maybe:** That project used constrained decoding for function selection. Here you’re doing free‑form generation — focus heavily on prompt quality to keep the 0.6B model on track.

---

## 📊 Phase 4: Evaluation & Self‑Testing (Day 9)

*Goal: Measure your recall@k and iterate until you meet the required thresholds.*

| # | Task | Description |
|---|------|-------------|
| 4.1 | **CLI: `evaluate`** | Compare your search results against the ground‑truth dataset. |
| 4.2 | **Recall@k calculation** | For each question, compute the share of correct sources retrieved in the top‑k. |
| 4.3 | **Overlap check** | A source counts as correct if it has the same `file_path` and an Intersection‑over‑Union (IoU) ≥ 0.05 with the reference span. |
| 4.4 | **Iterate** | Tune chunk size, chunking strategy, and retrieval method until you hit ≥80% recall@5 on docs and ≥50% on code. |

> ⚠️ **Important:** The official recall is computed by the **moulinette** executable during defence — your `evaluate` command is only for your own iteration and debugging.

---

## 📄 Phase 5: Documentation & Polish (Day 10)

*Goal: Complete README, ensure all linting passes, and produce a clean, professional deliverable.*

| # | Task | Description |
|---|------|-------------|
| 5.1 | **README first line** | Italicised: `*This project has been created as part of the 42 curriculum by [...]*` |
| 5.2 | **Description** | Explain the project goal and high‑level overview. |
| 5.3 | **Instructions** | Provide clear installation and execution steps. |
| 5.4 | **Resources** | List references and describe AI usage (if any). |
| 5.5 | **System architecture** | Include a diagram or textual description of the RAG pipeline components. |
| 5.6 | **Chunking strategy** | Detail your approach for Python vs Markdown chunking. |
| 5.7 | **Retrieval method** | Explain TF‑IDF/BM25 and your ranking mechanism. |
| 5.8 | **Performance analysis** | Discuss recall@k scores and overall system performance. |
| 5.9 | **Design decisions** | Justify key implementation choices. |
| 5.10 | **Challenges faced** | Describe difficulties and how you overcame them. |
| 5.11 | **Example usage** | Show clear examples of running the system. |
| 5.12 | **Linting** | `make lint` must pass (flake8 + mypy --strict). |
| 5.13 | **Type checking** | All functions must pass `mypy` strict checks. |
| 5.14 | **Docstrings** | PEP 257 style for all functions and classes. |

---

## 🏆 Phase 6: Bonus Features (Optional — 1 point each)

*Bonuses are graded **only after the entire mandatory part is validated**. Implement them in any order.*

### Bonus 1 – Semantic Embeddings
Add a vector index using a lightweight CPU model like `all-MiniLM-L6-v2`.

**Implementation tips:**
- Use the `sentence-transformers` library.
- Embed all chunks and store them in a FAISS index.
- Embed the query and search FAISS for the top‑k.
- Persist the vector index alongside the lexical index.

### Bonus 2 – Hybrid Retrieval
Combine lexical (TF‑IDF/BM25) and semantic rankings into a single result list.

**Implementation tips:**
- Retrieve top‑k from both indices.
- Normalise scores (e.g., Min‑Max scaling).
- Merge using Reciprocal Rank Fusion (RRF) or a weighted sum.
- Experiment with different fusion weights.

### Bonus 3 – Incremental Indexing
When a file changes, re‑index only that file instead of rebuilding the entire index.

**Implementation tips:**
- Store file modification timestamps or content hashes.
- On the `index` command, check which files changed.
- Remove old chunks for changed files and add new ones incrementally.

### Bonus 4 – Caching
Cache the index and query results to speed up cold starts and repeated queries.

**Implementation tips:**
- Use `functools.lru_cache` for query results.
- Persist the index to disk (already done in mandatory).
- Cache model responses for identical questions.
- Use `joblib` or `pickle` for serialisation.

### Bonus 5 – Local HTTP API
Expose querying and answering over a small local HTTP API.

**Implementation tips:**
- Use `FastAPI` or `Flask`.
- Endpoints: `/search`, `/answer`, `/index` (if incremental).
- Return JSON responses matching your Pydantic models.
- Run on `localhost:8000` with `uvicorn`.

---

## 📅 Project Timeline Summary

| Phase | Days | Key Deliverable |
|-------|------|-----------------|
| **Phase 0: Setup** | 1‑2 | Project skeleton, dependencies, Pydantic models. |
| **Phase 1: Indexing** | 3‑4 | Working indexer with TF‑IDF/BM25, ≤5 min indexing. |
| **Phase 2: Retrieval** | 5‑6 | Single & batch search, recall above thresholds. |
| **Phase 3: Answer Generation** | 7‑8 | Model integration, grounded answers. |
| **Phase 4: Evaluation** | 9 | Self‑testing, iteration to meet recall targets. |
| **Phase 5: Documentation** | 10 | README, linting, final polish. |
| **Phase 6: Bonuses** | +2‑3 | Optional features for extra points (1‑5). |

---

## 🚀 Critical Success Factors

1. **Chunking is everything** – Bad chunks lead to bad retrieval. Experiment with different strategies for Python (functions/classes) vs Markdown (headings/paragraphs).

2. **File paths must be exact** – The grader compares paths verbatim. Always use the exact paths from the ingested corpus.

3. **Stay under 2000 characters** – Any retrieved source longer than 2000 characters invalidates your entire output.

4. **Small model = limited reasoning** – `Qwen/Qwen3-0.6B` has known limitations. Focus on retrieval quality and prompt engineering rather than expecting miracles from the model.

5. **Follow the Call‑Me‑Maybe pattern** – That project used constrained decoding. Consider constraining your output format with structured prompts or JSON mode.

6. **Test early, test often** – Use the `evaluate` command to iterate on your retrieval strategy before generating answers.

7. **CPU‑only environment** – All bonuses must run on a CPU‑only campus machine. Keep models lightweight and avoid GPU dependencies.

---

## ⚙️ Consolidated CLI Commands

```bash
# Index the corpus
uv run python -m src index --max_chunk_size 2000

# Search a single query
uv run python -m src search "How does vLLM handle attention?" --k 10

# Search an entire dataset
uv run python -m src search_dataset --dataset_path data/questions.json --k 10 --save_directory data/output/

# Generate an answer for a single query
uv run python -m src answer "How does vLLM handle attention?" --k 10

# Generate answers for a dataset
uv run python -m src answer_dataset --student_search_results_path data/output/search_results.json --save_directory data/output/

# Evaluate your retrieval performance (self-testing)
uv run python -m src evaluate --predicted_path data/output/search_results.json --ground_truth_path data/ground_truth.json