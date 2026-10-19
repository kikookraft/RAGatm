"""Command-line entry point for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import fire
from tqdm import tqdm

from src.indexer import Indexer
from src.logging_config import configure_logging
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

configure_logging(level=logging.INFO)


def _missing_implementation(command_name: str) -> None:
    raise NotImplementedError(
        f"The '{command_name}' command is not implemented yet. "
        "Hook it up to the pipeline modules before running the CLI."
    )


def _read_json(path: str) -> Any:
    """Read and parse JSON from disk while handling malformed inputs."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    """Serialize payload to JSON on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _load_dataset(dataset_path: str) -> RagDataset:
    """Load and validate a RAG dataset from a JSON file."""
    if not isinstance(dataset_path, str) or (
            not dataset_path.strip()):
        raise ValueError("dataset_path must be a non-empty string")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")
    dataset = _read_json(dataset_path)
    return RagDataset.model_validate(dataset)


@lru_cache(maxsize=1)
def _get_llm() -> Any:
    """Lazily import and cache the LLM wrapper.

    Importing transformers/torch at module import time slows down every
    CLI command, including retrieval-only commands.
    """
    logging.info("Initializing LLM module...")
    from src.llm import llm as loaded_llm

    return loaded_llm


def _format_source_context(source: MinimalSource, text: str) -> str:
    """Format one retrieved chunk so the prompt keeps its provenance."""
    return (
        f"Source: {source.file_path}\n"
        f"Span: {source.first_character_index}-{source.last_character_index}\n"
        f"Content:\n{text.strip()}"
    )


@dataclass
class CLI:
    """Fire command surface for the project."""

    def index(self,
              max_chunk_size: int = 2000,
              directory: str = "data/raw") -> None:
        """Build the BM25 index and persist it under data/processed/."""
        if type(max_chunk_size) is not int or max_chunk_size <= 0:
            logging.error("max_chunk_size must be a positive integer.")
            return
        if type(directory) is not str:
            logging.error("directory must be a string.")
            return
        if os.path.exists(directory) is False:
            logging.error("directory does not exist.")
            return
        logging.info("Indexing with max_chunk_size=%d...", max_chunk_size)
        try:
            indexer = Indexer(directory, max_chunk_size)
            indexed = indexer.build_index()
            indexer.save()
            logging.info("Indexed %d chunks.", indexed)
        except KeyboardInterrupt:
            logging.warning(
                "Indexing interrupted — nothing was written. "
                "Re-run to try again."
            )
            raise SystemExit(130) from None
        except Exception as e:
            logging.error("Indexing failed: %s", e)
        return

    def search(self, query: str, k: int = 5) -> Any:
        try:
            if type(query) is not str:
                logging.error("query must be a string.")
                return
            if type(k) is not int or k <= 0:
                logging.error("k must be a positive integer.")
                return
            if k < 10:
                logging.warning("k is less than 10; consider using a larger ",
                                "value for better results.")
        except Exception as e:
            logging.error("Error in search parameters: %s", e)
            return
        try:
            indexer = Indexer.load()
            indexer.print_best_results(query, k)
        except json.JSONDecodeError:
            logging.error("Failed to load index, try make "
                          "index first then retry the command")
        except Exception as e:
            logging.error("Search failed: %s", e)
        return

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search every question in a dataset and write a JSON result file."""
        try:
            if not isinstance(dataset_path, str) or (
                    not dataset_path.strip()):
                raise ValueError("dataset_path must be a non-empty string")
            if not os.path.isfile(dataset_path):
                raise FileNotFoundError(
                    f"dataset file not found: {dataset_path}")
            if not isinstance(k, int) or k <= 0:
                raise ValueError("k must be a positive integer")

            dataset = _load_dataset(dataset_path)
            indexer = Indexer.load()
            results: list[MinimalSearchResults] = []

            for question in tqdm(
                dataset.rag_questions,
                desc="Searching dataset",
                unit="q",
                file=sys.stderr,
                dynamic_ncols=True,
                miniters=1,
                leave=True,
            ):
                retrieved = indexer.search(question.question, k, False)
                sources = [
                    MinimalSource(
                        file_path=source.file_path,
                        first_character_index=source.first_character_index,
                        last_character_index=source.last_character_index,
                    )
                    for source, _ in retrieved
                ]
                results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources,
                    )
                )

            payload = StudentSearchResults(search_results=results, k=k)
            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / Path(dataset_path).name
            _write_json(output_path, payload.model_dump(mode="json"))
            logging.info("Saved student_search_results to %s", output_path)
            return None
        except KeyboardInterrupt:
            logging.warning("Dataset search interrupted.")
            raise SystemExit(130) from None
        except Exception as exc:  # pragma: no cover - CLI guard
            logging.error("search_dataset failed: %s", exc)
            return None

    def answer(self, query: str, k: int = 10) -> Any:
        """Answer a single query using the top-k retrieved passages."""
        if not isinstance(query, str) or not query.strip():
            logging.error("query must be a non-empty string.")
            return None
        if not isinstance(k, int) or k <= 0:
            logging.error("k must be a positive integer.")
            return None
        try:
            indexer = Indexer.load()
            search_results = indexer.search(query, k)
            if not search_results:
                logging.warning("No sources found for query: %s", query)
                return {"question": query,
                        "answer": "No relevant source context was found.",
                        "retrieved_sources": []}

            sources: list[str] = []
            minimal_sources: list[MinimalSource] = []
            for source, _score in search_results:
                minimal_sources.append(source)
                try:
                    source_text = source.get_source_text()
                    sources.append(_format_source_context(source, source_text))
                except Exception as exc:
                    logging.warning("Failed to read source %s: %s",
                                    source.file_path, exc)

            answer = _get_llm().answer(query, sources)
            logging.info("Generated answer for query: %s", query)
            return {
                "question": query,
                # "retrieved_sources": [
                #     source.model_dump(mode="json") for source in minimal_sources
                # ],
                "answer": answer,
                # "full_sources": sources,
            }
        except Exception as exc:
            logging.error("answer failed: %s", exc)
            return None

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for every result in a
        StudentSearchResults JSON file."""
        try:
            if not isinstance(student_search_results_path, str
                              ) or not student_search_results_path.strip():
                raise ValueError(
                    "student_search_results_path must be a non-empty string")
            if not os.path.isfile(student_search_results_path):
                raise FileNotFoundError(
                    "search results file not "
                    f"found: {student_search_results_path}")

            with open(student_search_results_path,
                      "r", encoding="utf-8") as handle:
                payload = StudentSearchResults.model_validate(json.load(handle))

            answers: list[MinimalAnswer] = []
            for result in tqdm(
                payload.search_results,
                desc="Generating answers",
                unit="q",
                file=sys.stderr,
                dynamic_ncols=True,
                miniters=1,
                leave=True,
            ):
                source_texts: list[str] = []
                for source in result.retrieved_sources:
                    try:
                        source_text = source.get_source_text()
                        source_texts.append(
                            _format_source_context(source, source_text)
                        )
                    except Exception as exc:
                        logging.warning("Skipping invalid source %s: %s",
                                        source.file_path, exc)
                answer_text = _get_llm().answer(result.question, source_texts)
                answers.append(
                    MinimalAnswer(
                        question_id=result.question_id,
                        question=result.question,
                        retrieved_sources=result.retrieved_sources,
                        answer=answer_text,
                    )
                )

            output_payload = StudentSearchResultsAndAnswer(
                search_results=answers, k=payload.k)
            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / Path(student_search_results_path).name
            _write_json(output_path, output_payload.model_dump(mode="json"))
            logging.info("Saved student_search_results_and_answer to %s",
                         output_path)
            return None
        except KeyboardInterrupt:
            logging.warning("Answer generation interrupted.")
            raise SystemExit(130) from None
        except Exception as exc:  # pragma: no cover - CLI guard
            logging.error("answer_dataset failed: %s", exc)
            return None

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        _ = student_search_results_path
        _ = dataset_path
        _ = k
        _missing_implementation("evaluate")


def main() -> None:
    """Run the Fire command dispatcher.

    Installs a last-resort ``SIGINT`` handler so that a Ctrl-C landing
    anywhere in the pipeline — including inside third-party code such as
    ``bm25s`` or ``tqdm`` — exits with a short message and status code
    130 instead of an unhandled ``KeyboardInterrupt`` traceback.
    """
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        logging.warning("\nInterrupted — shutting down.")
        raise SystemExit(130) from None
    except Exception as e:
        logging.error("Error: %s", e)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
