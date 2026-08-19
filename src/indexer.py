# import bm25s
from tqdm import tqdm
from chunker import ChunkerFactory
from models import MinimalSource
from typing import List
from src.chunker import BaseChunker
# docs in https://github.com/xhluca/bm25s


class ChunkCollector:
    """Class that collect chunks of text and store them in a list."""
    def __init__(self, directory: str, max_chunk_size: int) -> None:
        self.chunks: list[MinimalSource] = []
        self.directory: str = directory
        self.max_chunk_size: int = max_chunk_size

    def add_chunk(self, chunk: MinimalSource) -> None:
        """Add a chunk of text to the list."""
        self.chunks.append(chunk)

    def get_chunks(self) -> list[MinimalSource]:
        """Return the list of chunks."""
        return self.chunks

    def collect_chunks(self) -> None:
        """Collect chunks of text from files in the directory."""
        from src.files_crawler import discovery

        file_paths: list[str] = discovery(self.directory)
        for file_path in tqdm(file_paths,
                              desc="Collecting chunks",
                              unit="file",
                              colour="cyan"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text: str = file.read()
                    chunker: BaseChunker = ChunkerFactory.get_chunker(
                        text, self.max_chunk_size)
                    chunks: List[MinimalSource] = chunker.chunk(
                        file_path, text)
                    for chunk in chunks:
                        self.add_chunk(chunk)
            except Exception as e:
                print(f"\033[91mError reading {file_path}: {e}\033[0m")
