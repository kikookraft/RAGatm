"""Indexing pipeline: chunk the corpus and build a persisted BM25 index.

The :class:`Indexer` walks the raw corpus, splits every file into chunks
no wider than ``max_chunk_size`` characters, and builds a BM25 lexical
index (https://github.com/xhluca/bm25s) that can be persisted under
``data/processed/`` and queried in milliseconds.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import bm25s
import Stemmer
from tqdm import tqdm

from src.chunker import _COLOR_RED, _COLOR_RESET, BaseChunker, ChunkerFactory
from src.files_crawler import discovery
from src.models import MinimalSource

logger = logging.getLogger(__name__)

#: The moulinette rejects any retrieved source longer than this many
#: characters, so chunking must never produce wider chunks.
#: -1 to disable the limit
MAX_CONTEXT_LENGTH: int = 2000
#: Default directory where the index is persisted.
DEFAULT_INDEX_DIRECTORY: str = "data/processed"
#: Default directory of the raw corpus.
DEFAULT_CORPUS_DIRECTORY: str = "data/raw"
#: Default number of results returned by a search.
DEFAULT_K: int = 10


def _validate_directory(directory: str) -> None:
    """Raise ``ValueError`` unless *directory* is an existing folder."""
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("directory must be a non-empty string")
    if not os.path.isdir(directory):
        raise ValueError(f"directory does not exist: {directory}")


def _validate_chunk_size(max_chunk_size: int) -> int:
    """Validate *max_chunk_size* and clamp it to ``MAX_CONTEXT_LENGTH``.

    Raises:
        ValueError: If *max_chunk_size* is not a positive integer.
    """
    if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if MAX_CONTEXT_LENGTH > 0 and max_chunk_size > MAX_CONTEXT_LENGTH:
        logger.warning(
            "max_chunk_size=%d exceeds the moulinette limit of %d — "
            "clamping to %d.",
            max_chunk_size,
            MAX_CONTEXT_LENGTH,
            MAX_CONTEXT_LENGTH,
        )
        return MAX_CONTEXT_LENGTH
    if MAX_CONTEXT_LENGTH > 0 and max_chunk_size < 20:
        logger.warning(
            "max_chunk_size=%d is very small — this may produce too many "
            "chunks and slow down indexing.",
            max_chunk_size,
        )
    return max_chunk_size


def _validate_k(k: int) -> int:
    """Validate *k*, treating invalid or negative values as ``0``."""
    if not isinstance(k, int) or k < 0:
        logger.warning("Invalid k=%r — falling back to k=0.", k)
        return 0
    return k


class ChunkCollector:
    """Walk a directory and collect chunk spans for every indexable file."""

    def __init__(self, directory: str, max_chunk_size: int) -> None:
        """Prepare an empty collection for *directory*.

        Raises:
            ValueError: If *directory* does not exist or *max_chunk_size*
                is not a positive integer.
        """
        _validate_directory(directory)
        self.directory: str = directory
        self.max_chunk_size: int = _validate_chunk_size(max_chunk_size)
        self.chunks: List[MinimalSource] = []
        self._text_cache: Dict[str, str] = {}

    def add_chunk(self, chunk: MinimalSource) -> None:
        """Append an already-built chunk to the collection."""
        self.chunks.append(chunk)

    def get_chunks(self) -> List[MinimalSource]:
        """Return all collected chunks."""
        return self.chunks

    def _read_file(self, file_path: str) -> Optional[str]:
        """Read *file_path* as UTF-8, caching its text.

        Returns:
            The file content, or ``None`` if it cannot be read.
        """
        if file_path in self._text_cache:
            return self._text_cache[file_path]
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text: str = file.read()
        except (OSError, UnicodeDecodeError) as exc:
            tqdm.write(
                f"{_COLOR_RED}Skipping {file_path}: {exc}{_COLOR_RESET}"
            )
            return None
        self._text_cache[file_path] = text
        return text

    def collect_chunks(self) -> None:
        """Chunk every discoverable file and store the resulting spans."""
        self.chunks = []
        self._text_cache = {}
        try:
            for file_path in tqdm(
                discovery(self.directory),
                desc="Collecting chunks",
                unit="file",
                colour="cyan",
            ):
                text = self._read_file(file_path)
                if text is None:
                    continue
                try:
                    chunker: BaseChunker = ChunkerFactory.get_chunker(
                        file_path, self.max_chunk_size
                    )
                    for chunk in chunker.chunk(file_path, text):
                        self.add_chunk(chunk)
                except Exception as exc:  # one bad file must not kill the run
                    tqdm.write(
                        f"{_COLOR_RED}Error chunking {file_path}: "
                        f"{exc}{_COLOR_RESET}"
                    )
        except KeyboardInterrupt:
            self.chunks = []
            self._text_cache = {}
            logger.info("Chunk collection cancelled.")
            raise

    def fetch_chunk_text(self, chunk: MinimalSource) -> str:
        """Return the text covered by *chunk*, from cache when possible."""
        cached = self._text_cache.get(chunk.file_path)
        if cached is not None:
            return cached[
                chunk.first_character_index:chunk.last_character_index
            ]
        try:
            return chunk.get_source_text()
        except Exception as exc:
            tqdm.write(
                f"{_COLOR_RED}Error fetching text for {chunk.file_path}: "
                f"{exc}{_COLOR_RESET}"
            )
            return ""

    def fetch_all(self) -> List[str]:
        """Return the text of every chunk, in collection order."""
        return [
            self.fetch_chunk_text(chunk)
            for chunk in tqdm(
                self.chunks,
                desc="Fetching chunk texts",
                unit="chunk",
                colour="green",
            )
        ]


class Indexer:
    """Build, persist and query a BM25 index over the chunked corpus."""

    def __init__(
        self,
        directory: str,
        max_chunk_size: int = MAX_CONTEXT_LENGTH,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> None:
        """Prepare an indexer for the corpus rooted at *directory*.

        Raises:
            ValueError: If *directory* does not exist or *max_chunk_size*
                is not a positive integer.
        """
        _validate_directory(directory)
        self.directory: str = directory
        self.max_chunk_size: int = _validate_chunk_size(max_chunk_size)
        self.index_directory: str = index_directory
        self.chunk_collector: ChunkCollector = ChunkCollector(
            directory, self.max_chunk_size
        )
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever: bm25s.BM25 = bm25s.BM25()
        self._is_indexed: bool = False

    def build_index(self) -> int:
        """Chunk the corpus, tokenize it, and fit the BM25 model.

        Returns:
            The number of indexed chunks.
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
        self, query: str, k: int = DEFAULT_K
    ) -> List[Tuple[MinimalSource, float]]:
        """Return the top-*k* chunks for *query* with their BM25 scores.

        Degenerate inputs (empty query, ``k=0``) yield an empty result
        list instead of raising.
        """
        logger.info("Searching for query: %r with k=%d", query, k)
        k = _validate_k(k)
        if k == 0:
            return []
        if not isinstance(query, str) or not query.strip():
            logger.warning("Empty query — returning no results.")
            return []
        if not self._is_indexed:
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

        Returns:
            The directory the index was written to.

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
