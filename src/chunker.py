from __future__ import annotations
import ast
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from src.files_crawler import discovery
from src.models import MinimalSource

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
_COLOR_RED = "\033[91m"
_COLOR_YELLOW = "\033[93m"
_COLOR_RESET = "\033[0m"

MAX_CONTEXT_LENGTH: int = 2000

# File extensions that map to each chunker strategy
_PYTHON_EXTENSIONS: frozenset[str] = frozenset({".py", ".pyw", ".pyx", ".pxd"})
_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".rst", ".txt",
     ".cfg", ".ini", ".toml", ".yaml", ".yml"}
)
# Source/config files worth indexing with the plain-text fallback
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {".c", ".h", ".cpp", ".cc", ".hpp", ".cu", ".cuh",
     ".sh", ".bash", ".cmake", ".json"}
)


def validate_directory(directory: str) -> None:
    """Raise ``ValueError`` unless *directory* is an existing folder."""
    if not directory.strip():
        raise ValueError("directory must be a non-empty string")
    if not os.path.isdir(directory):
        raise ValueError(f"directory does not exist: {directory}")


def validate_chunk_size(max_chunk_size: int) -> int:
    """Validate *max_chunk_size* and clamp it to ``MAX_CONTEXT_LENGTH``.
    """
    if max_chunk_size <= 0:
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
    if max_chunk_size < 20:
        logger.warning(
            "max_chunk_size=%d is very small — this may produce too many "
            "chunks and slow down indexing.",
            max_chunk_size,
        )
    return max_chunk_size


class BaseChunker(ABC):
    """Abstract chunker: turns file content into a list of MinimalSource."""

    def __init__(
        self,
        max_chunk_size: int,
        target_chunk_size: int | None = None,
        min_chunk_size: int | None = None,
    ) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be a positive integer")
        self.max_chunk_size: int = max_chunk_size
        self.target_chunk_size: int = min(
            max_chunk_size,
            target_chunk_size if target_chunk_size is not None
            else max_chunk_size,
        )
        default_min = max(200, self.target_chunk_size // 2)
        self.min_chunk_size: int = max(
            200,
            min(self.max_chunk_size,
                min_chunk_size if min_chunk_size is not None else default_min),
        )

    def effective_limit(self) -> int:
        """Return the chunk-size limit used by this file-type strategy."""
        return min(self.max_chunk_size, self.target_chunk_size)

    @abstractmethod
    def chunk(self, file_path: str, data: str) -> List[MinimalSource]:
        """Split *data* into sources no wider than max_chunk_size.
        """
        ...

    @staticmethod
    def _build_source(
        file_path: str,
        start_char: int,
        end_char: int,
    ) -> Optional[MinimalSource]:
        """Create MinimalSource for a span,
        or return None if the span is empty.
        """
        if end_char <= start_char:
            logger.debug(
                "Skipping empty span %d:%d in %s",
                start_char,
                end_char,
                file_path,
            )
            return None
        return MinimalSource(
            file_path=file_path,
            first_character_index=start_char,
            last_character_index=end_char,
        )

    def _emit(
        self,
        sources: List[MinimalSource],
        file_path: str,
        start_char: int,
        end_char: int,
    ) -> None:
        """Append the source for a span to *sources*, skipping empty ones.
        """
        source = self._build_source(file_path, start_char, end_char)
        if source is not None:
            sources.append(source)

    @staticmethod
    def _absolute_offset(data: str, lineno: int, col_offset: int) -> int:
        """Convert an AST ``(lineno, col_offset)`` pair to a char offset.

        ``lineno`` is 1-based; ``col_offset`` counts characters from the
        start of the line.
        """
        lines = data.splitlines(keepends=True)
        line_index = min(lineno - 1, len(lines) - 1)
        offset = sum(len(line) for line in lines[:line_index])
        return offset + col_offset

    @staticmethod
    def _split_by_lines(
        data: str,
        start_offset: int,
        max_size: int,
    ) -> List[Tuple[int, int]]:
        """Split *data* into ``(start, end)`` byte-offset pairs at line
        boundaries, each no longer than *max_size*.
        This is the fallback strategy used when a single AST node exceeds
        ``max_chunk_size``.
        """
        spans: List[Tuple[int, int]] = []
        remaining = data
        cursor = start_offset

        while remaining:
            if len(remaining) <= max_size:
                spans.append((cursor, cursor + len(remaining)))
                break
            cut = remaining.rfind("\n", 0, max_size)
            if cut == -1:  # if no newline found, hard cut at max_size
                cut = max_size
            else:
                cut += 1  # include the newline in the current chunk
            spans.append((cursor, cursor + cut))
            cursor += cut
            remaining = remaining[cut:]
        return spans


class PythonChunker(BaseChunker):
    """Chunk Python source at top-level AST node boundaries.
    Split by lines if a single node exceeds *max_chunk_size*.
    """

    def __init__(self,
                 max_chunk_size: int,
                 file_path: str | None = None) -> None:
        super().__init__(
            max_chunk_size,
            target_chunk_size=min(max_chunk_size, 1200),
            min_chunk_size=max(300, min(max_chunk_size, 600)),
        )
        self.file_path = file_path

    def chunk(self, file_path: str, data: str) -> List[MinimalSource]:
        sources: List[MinimalSource] = []
        limit = self.effective_limit()

        try:
            tree = ast.parse(data)
        except SyntaxError:  # fallback for unparseable files
            tqdm.write(
                f"{_COLOR_YELLOW}Syntax error in {file_path} — falling "
                f"back to line splitting{_COLOR_RESET}"
            )
            for start, end in self._split_by_lines(
                data, 0, limit
            ):
                self._emit(sources, file_path, start, end)
            return sources

        # collect (start, end, text) tuples, then sort by start offset
        raw_segments: List[Tuple[int, int, str]] = []
        for node in tree.body:
            node_start = self._absolute_offset(data, node.lineno,
                                               node.col_offset)
            node_end = self._absolute_offset(
                data,
                node.end_lineno if node.end_lineno is not None else 0,
                node.end_col_offset or 0,
            )
            seg_text = data[node_start:node_end]
            if not seg_text.strip():
                continue

            if len(seg_text) <= limit:
                raw_segments.append((node_start, node_end, seg_text))
            else:
                logger.debug(
                    "Node at line %d in %s is %d chars — line-splitting",
                    node.lineno,
                    file_path,
                    len(seg_text),
                )
                for s, e in self._split_by_lines(
                    seg_text, node_start, limit
                ):
                    raw_segments.append((s, e, data[s:e]))

        if not raw_segments:
            return sources

        # Sort by start offset and remove overlaps
        raw_segments.sort(key=lambda seg: seg[0])
        segments: List[Tuple[int, int, str]] = []
        covered_until = 0
        for seg_start, seg_end, seg_text in raw_segments:
            if seg_end <= covered_until:
                continue  # fully inside a preceding segment
            if seg_start < covered_until:
                # Partial overlap — trim to avoid double-counting
                trimmed = data[covered_until:seg_end]
                segments.append((covered_until, seg_end, trimmed))
            else:
                segments.append((seg_start, seg_end, seg_text))
            covered_until = seg_end

        #  Group consecutive segments into chunks
        current_start: int = segments[0][0]
        current_end: int = segments[0][1]

        for seg_start, seg_end, _seg_text in segments[1:]:
            if (seg_end - current_start) <= limit:
                current_end = seg_end
            else:
                self._emit(
                    sources,
                    file_path,
                    current_start,
                    current_end,
                )
                current_start = seg_start
                current_end = seg_end

        self._emit(
            sources,
            file_path,
            current_start,
            current_end,
        )

        return sources


class TextChunker(BaseChunker):
    """Chunk prose documents (`.md`, `.rst`, `.txt`) at structural
    Split first by double newlines, then by punctuation, then by single spaces.
    """

    def __init__(self, max_chunk_size: int) -> None:
        super().__init__(
            max_chunk_size,
            target_chunk_size=min(max_chunk_size, 1500),
            min_chunk_size=max(300, min(max_chunk_size, 600)),
        )

    def chunk(self, file_path: str, data: str) -> List[MinimalSource]:
        sources: List[MinimalSource] = []
        remaining = data
        cursor = 0  # absolute character offset within *data*
        limit = self.effective_limit()

        while remaining:
            if len(remaining) <= limit:
                self._emit(
                    sources, file_path, cursor,
                    cursor + len(remaining)
                )
                break

            cut = remaining.rfind("\n\n", 0, limit)
            if cut != -1:
                cut += 2  # include both newlines
            else:
                cut = -1
                for punct in (". ", "? ", "! ", ".\n", "?\n", "!\n", ":\n"):
                    pos = remaining.rfind(punct, 0, limit)
                    if pos != -1:
                        cut = max(cut, pos + len(punct))

            if cut == -1:
                cut = remaining.rfind(" ", 0, limit)

            if cut <= 0:  # no suitable split point found; hard cut at limit
                cut = limit

            cut = min(cut, limit)  # force cut to not exceed limit

            self._emit(
                sources,
                file_path,
                cursor,
                cursor + cut,
            )
            cursor += cut
            remaining = remaining[cut:]

        return sources


class PlainTextChunker(BaseChunker):
    """Chunk any file generically at line boundaries.

    Used for extensions that have no dedicated strategy (C/C++/CUDA
    sources, shell scripts, JSON, ...). Falls back to hard character
    splits for very long lines.
    """

    def __init__(self, max_chunk_size: int) -> None:
        super().__init__(
            max_chunk_size,
            target_chunk_size=min(max_chunk_size, 1000),
            min_chunk_size=max(250, min(max_chunk_size, 400)),
        )

    def chunk(self, file_path: str, data: str) -> List[MinimalSource]:
        sources: List[MinimalSource] = []
        limit = self.effective_limit()
        for start, end in self._split_by_lines(data, 0, limit):
            self._emit(sources, file_path, start, end)
        return sources


class ChunkerFactory:
    """Select the appropriate chunker for a given file path."""

    @staticmethod
    def get_chunker(file_path: str, max_chunk_size: int) -> BaseChunker:
        """Return a chunker instance for *file_path*.

        Args:
            file_path: Absolute or relative path to the file.
            max_chunk_size: Maximum character width of each chunk.

        Returns:
            A :class:`BaseChunker` subclass instance. Unknown
            extensions get the :class:`PlainTextChunker` fallback.
        """
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()

        if ext_lower in _PYTHON_EXTENSIONS:
            return PythonChunker(max_chunk_size)
        if ext_lower in _TEXT_EXTENSIONS:
            return TextChunker(max_chunk_size)
        # if ext_lower in _CODE_EXTENSIONS:
        #     return PlainTextChunker(max_chunk_size)

        logger.debug(
            "No dedicated chunker for '%s' — using plain-text fallback",
            file_path,
        )
        return PlainTextChunker(max_chunk_size)


class ChunkCollector:
    """Walk a directory and collect chunk spans for every indexable file."""

    def __init__(self, directory: str, max_chunk_size: int) -> None:
        """Prepare an empty collection for *directory*.
        Raise value errors if *directory* does not exist
        """
        validate_directory(directory)
        self.directory: str = directory
        self.max_chunk_size: int = validate_chunk_size(max_chunk_size)
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
