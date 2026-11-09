from __future__ import annotations
import json
import logging
import os
from typing import List, Optional, Tuple
import bm25s
import Stemmer
from src.chunker import (ChunkCollector, validate_chunk_size,
                         validate_directory)
from src.models import MinimalSource

logger = logging.getLogger(__name__)


MAX_CONTEXT_LENGTH: int = 2000
DEFAULT_INDEX_DIRECTORY: str = "data/processed"
DEFAULT_CORPUS_DIRECTORY: str = "data/raw"
DEFAULT_K: int = 10


def _validate_k(k: int) -> int:
    """Validate *k*, treating invalid or negative values as ``0``."""
    if k < 0:
        logger.warning("Invalid k=%r — falling back to k=0.", k)
        return 0
    return k


class Indexer:
    """Build, persist and query a BM25 index over the chunked corpus."""

    def __init__(
        self,
        directory: str,
        max_chunk_size: int = 2000,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> None:
        """Prepare an indexer for the corpus rooted at *directory*.
        """
        validate_directory(directory)
        self.directory: str = directory
        self.max_chunk_size: int = validate_chunk_size(max_chunk_size)
        self.index_directory: str = index_directory
        self.chunk_collector: ChunkCollector = ChunkCollector(
            directory, self.max_chunk_size
        )
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever: bm25s.BM25 = bm25s.BM25()
        self._is_indexed: bool = False

    def build_index(self) -> int:
        """Chunk the corpus, tokenize it, and fit the BM25 model.
        Returns: The number of indexed chunks.
        """
        logger.info("Collecting and tokenizing the corpus...")
        self._is_indexed = False
        self.chunk_collector.collect_chunks()
        corpus: List[str] = self.chunk_collector.fetch_all()
        if not corpus:
            logger.warning("No chunks collected — nothing to index.")
            return 0
        corpus_tokens = bm25s.tokenize(
            corpus, stopwords="en", stemmer=self.stemmer
        )
        logger.info("Building BM25 index over %d chunks...", len(corpus))
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self._is_indexed = True
        return len(corpus)

    def search(
        self, query: str, k: int = DEFAULT_K, log_results: bool = True
    ) -> List[Tuple[MinimalSource, float]]:
        """Return the top-*k* chunks for *query* with their BM25 scores.

        Degenerate inputs (empty query, ``k=0``) yield an empty result
        list instead of raising.
        """
        if log_results:
            logger.info("Searching for query: %r with k=%d", query, k)
        k = _validate_k(k)
        if k == 0:
            return []
        if not query.strip():
            logger.warning("Empty query — returning no results.")
            return []
        if not self._is_indexed:
            if log_results:
                logger.warning("Index not built — building it now...")
            self.build_index()
            # raise RuntimeError("Index not built — call build_index() first.")

        query_tokens = bm25s.tokenize(
            query, stopwords="en", stemmer=self.stemmer
        )
        top_k = min(k, len(self.chunk_collector.chunks))
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)
        chunks = self.chunk_collector.get_chunks()
        return [
            (chunks[doc_id], float(score))
            for doc_id, score in zip(results[0].tolist(), scores[0].tolist())
        ]

    def print_best_results(self, query: str, k: int = DEFAULT_K) -> None:
        """Print the top-*k* chunks for *query* with their BM25 scores."""
        results = self.search(query, k)
        if not results:
            print("No results found.")
            return
        print(f"Top {len(results)} results for query: {query}")
        for idx, (chunk, score) in enumerate(results, start=1):
            print(f"{idx}. Score: {score:.4f}, Chunk: {chunk}")

    def save(self, index_directory: Optional[str] = None) -> str:
        """Persist the BM25 index and chunk metadata under a directory.
        If *index_directory* is not provided, the default directory
        specified at initialization is used.
        Returns: The path to the directory where the index was saved.
        Raises:
            RuntimeError: If no complete index has been built yet.
        """
        if not self._is_indexed:
            # Refuse to write a truncated index: load() would then happily
            # read back a corpus missing most of its chunks.
            raise RuntimeError(
                "Refusing to save: no complete index has been built."
            )
        target: str = index_directory or self.index_directory
        os.makedirs(target, exist_ok=True)
        self.retriever.save(os.path.join(target, "bm25"))
        chunks_path = os.path.join(target, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as file:
            json.dump(
                [c.model_dump() for c in self.chunk_collector.get_chunks()],
                file,
            )
        logger.info("Index saved under %s", target)
        return target

    @classmethod
    def load(
        cls,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        corpus_directory: str = DEFAULT_CORPUS_DIRECTORY,
    ) -> "Indexer":
        """Restore an index previously written by :meth:`save`.

        Raises:
            ValueError: If no saved index exists under *index_directory*.
        """
        bm25_path = os.path.join(index_directory, "bm25")
        chunks_path = os.path.join(index_directory, "chunks.json")
        if not os.path.isdir(bm25_path) or not os.path.isfile(chunks_path):
            raise ValueError(
                f"No saved index found under: {index_directory}"
            )
        instance = cls(corpus_directory, index_directory=index_directory)
        instance.retriever = bm25s.BM25.load(bm25_path)
        try:
            with open(chunks_path, "r", encoding="utf-8") as file:
                instance.chunk_collector.chunks = [
                    MinimalSource(**item) for item in json.load(file)
                ]
        except json.JSONDecodeError as e:
            logger.error("Error decoding chunks.json: %s", e)
            raise
        except Exception as e:
            logger.error("Error loading chunks: %s", e)
            raise
        instance._is_indexed = True
        logger.info(
            "Loaded index with %d chunks from %s",
            len(instance.chunk_collector.chunks),
            index_directory,
        )
        return instance


if __name__ == "__main__":
    indexer = Indexer(DEFAULT_CORPUS_DIRECTORY, MAX_CONTEXT_LENGTH)
    indexed = indexer.build_index()
    indexer.save()
    print(f"Indexed {indexed} chunks.")

    user_query: str = input("Enter your query: ")
    try:
        for source, score in indexer.search(user_query, DEFAULT_K):
            print(f"Score: {score:.4f}, Chunk: {source}")
    except RuntimeError as exc:
        print(f"Search failed: {exc}")
