"""Command-line entry point for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fire
import logging
from src.indexer import Indexer
from src.logging_config import configure_logging
import os

configure_logging(level=logging.INFO)


def _missing_implementation(command_name: str) -> None:
    raise NotImplementedError(
        f"The '{command_name}' command is not implemented yet. "
        "Hook it up to the pipeline modules before running the CLI."
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
            # KeyboardInterrupt derives from BaseException, so the
            # `except Exception` below never sees it. Catch it explicitly
            # to report cleanly instead of dumping a traceback.
            logging.warning(
                "Indexing interrupted — nothing was written. "
                "Re-run to try again."
            )
            raise SystemExit(130) from None
        except Exception as e:
            logging.error("Indexing failed: %s", e)
        return

    def search(self, query: str, k: int = 5) -> Any:
        indexer = Indexer.load()
        indexer.print_best_results(query, k)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        _ = dataset_path
        _ = k
        _ = save_directory
        _missing_implementation("search_dataset")

    def answer(self, query: str, k: int = 10) -> Any:
        _ = query
        _ = k
        return _missing_implementation("answer")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        _ = student_search_results_path
        _ = save_directory
        _missing_implementation("answer_dataset")

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


if __name__ == "__main__":
    main()
