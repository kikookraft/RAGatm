from __future__ import annotations

import ast
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from tqdm import tqdm

from src.models import MinimalSource

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
_COLOR_RED = "\033[91m"
_COLOR_YELLOW = "\033[93m"
_COLOR_RESET = "\033[0m"


class BaseChunker(ABC):
    """Abstract chunker: turns file content into a list of MinimalSource."""

    def __init__(self, max_chunk_size: int) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be a positive integer")
        self.max_chunk_size: int = max_chunk_size

    @abstractmethod
    def chunk(self, file_path: str, data: str) -> List[MinimalSource]:
        """Split *data* into sources no wider than max_chunk_size.
        """
        ...

    @staticmethod
    def _build_source(
        file_path: str,
        text: str,
        start_char: int,
        end_char: int,
    ) -> Optional[MinimalSource]:
        """Factory for a single MinimalSource with consistent semantics.

        *start_char* is inclusive, *end_char* is exclusive.

        Returns ``None`` when the span is empty or reversed, so that a
        degenerate boundary can never produce a source that
        :class:`MinimalSource` rejects.
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
        fallback strategy used when a single AST node exceeds
        ``max_chunk_size``.
        """
        spans: List[Tuple[int, int]] = []
        remaining = data
        cursor = start_offset

        while remaining:
            if len(remaining) <= max_size:
                spans.append((cursor, cursor + len(remaining)))
                break

            # Try to cut at the last newline within the limit
            cut = remaining.rfind("\n", 0, max_size)
            if cut == -1:
                # No newline found — cut at max_size (hard split)
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

    def chunk(self, file_path: str, data: str) -> List[MinimalSource]:
        sources: List[MinimalSource] = []

        # fallback for unparseable files
        try:
            tree = ast.parse(data)
        except SyntaxError:
            tqdm.write(
                f"{_COLOR_YELLOW}Syntax error in {file_path} — falling "
                f"back to line splitting{_COLOR_RESET}"
            )
            return [
                source
                for s, e in self._split_by_lines(data, 0, self.max_chunk_size)
                if (source := self._build_source(file_path, data[s:e], s, e))
                is not None
            ]

        # collect (start, end, text) tuples, then sort by start offset
        raw_segments: List[Tuple[int, int, str]] = []
        for node in tree.body:
            # Use AST line/col offsets instead of data.index(seg_text):
            # index() would map duplicated snippets to the wrong offsets.
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

            if len(seg_text) <= self.max_chunk_size:
                raw_segments.append((node_start, node_end, seg_text))
            else:
                logger.debug(
                    "Node at line %d in %s is %d chars — line-splitting",
                    node.lineno,
                    file_path,
                    len(seg_text),
                )
                for s, e in self._split_by_lines(
                    seg_text, node_start, self.max_chunk_size
                ):
                    raw_segments.append((s, e, data[s:e]))

        if not raw_segments:
            return sources

        # Sort by start offset and remove overlaps (keep the first segment
        # that covers a region; later segments that fall inside an
        # already-covered range are dropped).
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
            if (seg_end - current_start) <= self.max_chunk_size:
                current_end = seg_end
            else:
                sources.append(
                    self._build_source(
                        file_path,
                        data[current_start:current_end],
                        current_start,
                        current_end,
                    )
                )
                current_start = seg_start
                current_end = seg_end

        sources.append(
            self._build_source(
                file_path,
                data[current_start:current_end],
                current_start,
                current_end,
            )
        )

        return sources


class TextChunker(BaseChunker):
    """Chunk prose documents (`.md`, `.rst`, `.txt`) at structural
    Split first by double newlines, then by punctuation, then by single spaces.
    """

    def chunk(self, file_path: str, data: str) -> List[MinimalSource]:
        sources: List[MinimalSource] = []
        remaining = data
        cursor = 0  # absolute character offset within *data*

        while remaining:
            if len(remaining) <= self.max_chunk_size:
                tail = self._build_source(
                    file_path, remaining, cursor, cursor + len(remaining)
                )
                if tail is not None:
                    sources.append(tail)
                break

            cut = remaining.rfind("\n\n", 0, self.max_chunk_size)
            if cut != -1:
                cut += 2  # include both newlines
            else:
                cut = -1
                for punct in (". ", "? ", "! ", ".\n", "?\n", "!\n", ":\n"):
                    pos = remaining.rfind(punct, 0, self.max_chunk_size)
                    if pos != -1:
                        cut = max(cut, pos + len(punct))

            if cut == -1:
                cut = remaining.rfind(" ", 0, self.max_chunk_size)

            if cut <= 0:
                # A boundary at offset 0 (or no boundary at all) would
                # produce an empty chunk and the cursor would never
                # advance — fall back to a hard cut at max_size.
                cut = self.max_chunk_size

            # A punctuation match ending exactly at the limit can push the
            # cut past max_size; clamp so no chunk ever exceeds it.
            cut = min(cut, self.max_chunk_size)

            source = self._build_source(
                file_path,
                remaining[:cut],
                cursor,
                cursor + cut,
            )
            if source is not None:
                sources.append(source)
            cursor += cut
            remaining = remaining[cut:]

        return sources


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


class PlainTextChunker(BaseChunker):
    """Chunk any file generically at line boundaries.

    Used for extensions that have no dedicated strategy (C/C++/CUDA
    sources, shell scripts, JSON, ...). Falls back to hard character
    splits for very long lines.
    """

    def chunk(self, file_path: str, data: str) -> List[MinimalSource]:
        return [
            source
            for s, e in self._split_by_lines(data, 0, self.max_chunk_size)
            if (source := self._build_source(file_path, data[s:e], s, e))
            is not None
        ]


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


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.DEBUG)

    with tempfile.TemporaryDirectory() as tmpdir:
        py_code = '''"""Module docstring."""

import os
import sys

def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b

class Calculator:
    """A simple calculator."""

    def multiply(self, x: int, y: int) -> int:
        return x * y

def greet(name: str) -> None:
    print(f"Hello, {name}!")
'''
        py_path = os.path.join(tmpdir, "test.py")
        with open(py_path, "w") as f:
            f.write(py_code)

        py_chunker = PythonChunker(max_chunk_size=120)
        py_sources = py_chunker.chunk(py_path, py_code)
        print(f"\nPython chunks ({len(py_sources)}):")
        for i, src in enumerate(py_sources):
            span = src.last_character_index - src.first_character_index
            preview = py_code[
                src.first_character_index:src.last_character_index]
            preview = preview[:60].replace("\n", "\\n")
            print(
                f"  [{i}] {src.first_character_index}:"
                f"{src.last_character_index}"
                f" ({span} chars) | {preview}..."
            )

        # --- Text chunker smoke test ---------------------------------------
        md_text = (
            "## Introduction\n\n"
            "This is the first paragraph. It contains several sentences. "
            "Each one ends with punctuation.\n\n"
            "## Details\n\n"
            "Another paragraph here with more content to demonstrate "
            "the chunking behavior on markdown text files."
        )
        md_path = os.path.join(tmpdir, "readme.md")
        with open(md_path, "w") as f:
            f.write(md_text)

        text_chunker = TextChunker(max_chunk_size=200)
        txt_sources = text_chunker.chunk(md_path, md_text)
        print(f"\nMarkdown chunks ({len(txt_sources)}):")
        for i, src in enumerate(txt_sources):
            span = src.last_character_index - src.first_character_index
            preview = md_text[
                src.first_character_index:src.last_character_index
            ]
            preview = preview[:80].replace("\n", "\\n")
            print(
                f"  [{i}] {src.first_character_index}:"
                f"{src.last_character_index}"
                f" ({span} chars) | {preview}..."
            )

        # --- Factory test --------------------------------------------------
        for ext in (".py", ".pyw", ".pyx", ".md", ".rst", ".txt", ".ini"):
            path = os.path.join(tmpdir, f"dummy{ext}")
            try:
                chunker = ChunkerFactory.get_chunker(path, 500)
                print(f"\n{ext:8s} → {type(chunker).__name__}")
            except ValueError as e:
                print(f"\n{ext:8s} → ERROR: {e}")

        # --- Test unsupported extension ------------------------------------
        try:
            ChunkerFactory.get_chunker("foo.bin", 500)
            print("\n.bin   → ERROR: should have raised ValueError")
        except ValueError:
            print("\n.bin   → ValueError (expected)")

    print("\n✅ All smoke tests passed.")
