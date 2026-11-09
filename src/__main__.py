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
from tqdm.contrib.logging import logging_redirect_tqdm

from src.indexer import Indexer
from src.logging_config import configure_logging
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

configure_logging(level=logging.INFO)


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


def _compute_span_iou(left: MinimalSource, right: MinimalSource) -> float:
    """Return the Intersection-over-Union of two source spans."""
    if left.file_path != right.file_path:
        return 0.0

    start = max(left.first_character_index, right.first_character_index)
    end = min(left.last_character_index, right.last_character_index)
    overlap = max(0, end - start)
    union_start = min(left.first_character_index, right.first_character_index)
    union_end = max(left.last_character_index, right.last_character_index)
    union = max(0, union_end - union_start)
    if union <= 0:
        return 0.0
    return overlap / union


def _source_matches_reference(
    candidate: MinimalSource,
    reference: MinimalSource,
) -> bool:
    """Check whether a candidate source covers the same file region."""
    return (
        candidate.file_path == reference.file_path
        and _compute_span_iou(candidate, reference) >= 0.05
    )


def _compute_recall_for_question(
    candidate_sources: list[MinimalSource],
    reference_sources: list[MinimalSource],
    limit: int,
) -> float:
    """Return recall@limit for a single question."""
    if not reference_sources:
        return 0.0
    if limit <= 0:
        return 0.0

    matched: set[int] = set()
    for reference_index, reference_source in enumerate(reference_sources):
        for candidate in candidate_sources[:limit]:
            if _source_matches_reference(candidate, reference_source):
                matched.add(reference_index)
                break
    return len(matched) / len(reference_sources)


def evaluate_student_search_results(
    student_results: Any,
    dataset: Any,
    k: int = 10,
) -> dict[str, Any]:
    """Validate search results and compute recall metrics against a dataset."""
    try:
        payload = (
            student_results
            if isinstance(student_results, StudentSearchResults)
            else StudentSearchResults.model_validate(student_results)
        )
    except Exception as exc:  # pragma: no cover - validation path
        return {
            "valid": False,
            "error": f"student results are invalid: {exc}",
            "dataset_metrics": {},
            "per_question": {},
        }

    try:
        dataset_payload = (
            dataset
            if isinstance(dataset, RagDataset)
            else RagDataset.model_validate(dataset)
        )
    except Exception as exc:  # pragma: no cover - validation path
        return {
            "valid": False,
            "error": f"dataset is invalid: {exc}",
            "dataset_metrics": {},
            "per_question": {},
        }

    metric_levels = (1, 3, 5, 10)
    per_question: dict[str, dict[str, float]] = {}
    question_scores: dict[str, list[float]] = {
        f"Recall@{level}": [] for level in metric_levels
    }

    for question in dataset_payload.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            continue

        expected_sources = list(question.sources)
        result = next(
            (
                item for item in payload.search_results
                if item.question_id == question.question_id
            ),
            None,
        )
        candidate_sources = (
            list(result.retrieved_sources)
            if result is not None
            else []
        )
        question_metrics: dict[str, float] = {}
        for level in metric_levels:
            eval_limit = min(level, max(payload.k, k)
                             ) if payload.k > 0 else level
            score = _compute_recall_for_question(
                candidate_sources,
                expected_sources,
                eval_limit,
            )
            question_metrics[f"recall@{level}"] = score
            question_scores[f"Recall@{level}"].append(score)
        per_question[question.question_id] = question_metrics

    dataset_metrics = {
        f"Recall@{level}": (
            sum(question_scores[f"Recall@{level}"]) / len(
                question_scores[f"Recall@{level}"])
            if question_scores[f"Recall@{level}"]
            else 0.0
        )
        for level in metric_levels
    }

    return {
        "valid": True,
        "error": None,
        "dataset_metrics": dataset_metrics,
        "per_question": per_question,
        "n_questions": len(dataset_payload.rag_questions),
    }


@lru_cache(maxsize=1)
def _get_llm() -> Any:
    """Lazily import and cache the LLM wrapper.
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
                "retrieved_sources": [
                    source.model_dump(mode="json") for source in minimal_sources
                ],
                "answer": answer,
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
            if not isinstance(save_directory, str) or not (
                    save_directory.strip()):
                raise ValueError(
                    "save_directory must be a non-empty string")
            if not os.path.isfile(student_search_results_path):
                raise FileNotFoundError(
                    "search results file not "
                    f"found: {student_search_results_path}")

            with open(student_search_results_path,
                      "r", encoding="utf-8") as handle:
                payload = StudentSearchResults.model_validate(json.load(handle))

            answers: list[MinimalAnswer] = []
            with logging_redirect_tqdm():
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
                    answer_text = _get_llm().answer(
                        result.question, source_texts
                    )
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
    ) -> dict[str, Any] | None:
        """Validate search results and print recall metrics for a dataset."""
        try:
            if not isinstance(student_search_results_path, str) or (
                    not student_search_results_path.strip()):
                raise ValueError(
                    "student_search_results_path must be a non-empty string"
                )
            if not os.path.isfile(student_search_results_path):
                raise FileNotFoundError(
                    f"student search results file not found: "
                    f"{student_search_results_path}"
                )
            if not isinstance(dataset_path, str) or not dataset_path.strip():
                raise ValueError("dataset_path must be a non-empty string")
            if not os.path.isfile(dataset_path):
                raise FileNotFoundError(
                    f"dataset file not found: {dataset_path}")
            if not isinstance(k, int) or k <= 0:
                raise ValueError("k must be a positive integer")

            with open(
                    student_search_results_path, "r",
                    encoding="utf-8") as handle:
                student_results = json.load(handle)
            with open(dataset_path, "r", encoding="utf-8") as handle:
                dataset = json.load(handle)

            result = evaluate_student_search_results(student_results,
                                                     dataset, k)
            print(f"Student data is valid: {result['valid']}")
            print("Evaluation Results")
            print("========================================")
            for level in (1, 3, 5, 10):
                metric_name = f"Recall@{level}"
                value = result["dataset_metrics"].get(metric_name, 0.0)
                print(f"{metric_name}: {value:.3f}")
            return result
        except KeyboardInterrupt:
            logging.warning("Evaluation interrupted.")
            raise SystemExit(130) from None
        except Exception as exc:  # pragma: no cover - CLI guard
            logging.error("evaluation failed: %s", exc)
            return {
                "valid": False,
                "error": str(exc),
                "dataset_metrics": {},
                "per_question": {},
            }


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
