from __future__ import annotations

from src.chunker import ChunkerFactory, PythonChunker, TextChunker


def test_chunker_factory_uses_file_type_targets() -> None:
    python_chunker = ChunkerFactory.get_chunker("demo.py", 2000)
    text_chunker = ChunkerFactory.get_chunker("demo.md", 2000)

    assert isinstance(python_chunker, PythonChunker)
    assert isinstance(text_chunker, TextChunker)
    assert python_chunker.target_chunk_size == 1200
    assert text_chunker.target_chunk_size == 1500
    assert python_chunker.max_chunk_size == 2000
    assert text_chunker.max_chunk_size == 2000
    assert python_chunker.min_chunk_size >= 300
    assert text_chunker.min_chunk_size >= 300


def test_python_chunker_respects_dynamic_target_limit() -> None:
    data = "def demo():\n" + "\n".join(f"value_{index} = {index}" for index in range(400))
    chunker = PythonChunker(2000)

    chunks = chunker.chunk("demo.py", data)

    assert chunks
    assert all(
        chunk.last_character_index - chunk.first_character_index <= 1200
        for chunk in chunks
    )
