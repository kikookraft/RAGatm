"""Command-line entry point for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fire
import logging
from src.indexer import Indexer
from src.logging_config import configure_logging

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
        logging.info("Indexing with max_chunk_size=%d...", max_chunk_size)
        indexer = Indexer(directory, max_chunk_size)
        indexed = indexer.build_index()
        indexer.save()
        logging.info("Indexed %d chunks.", indexed)

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
    """Run the Fire command dispatcher."""

    fire.Fire(CLI)


if __name__ == "__main__":
    main()
