# import bm25s
from tqdm import tqdm
from src.chunker import ChunkerFactory, _COLOR_RED, _COLOR_RESET
from src.models import MinimalSource
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
                tqdm.write(
                    f"{_COLOR_RED}Error reading {file_path}: {e}{_COLOR_RESET}"
                )

    def fetch_chunk_text(self, chunk: MinimalSource) -> str:
        """Fetch the text of a chunk from its source file."""
        try:
            return chunk.get_source_text()
        except Exception as e:
            tqdm.write(
                f"{_COLOR_RED}Error fetching text for {chunk}: "
                f"{e}{_COLOR_RESET}"
            )
            return ""

    def fetch_all(self) -> List[str]:
        """Fetch the text of all chunks."""
        return [self.fetch_chunk_text(chunk) for chunk in self.chunks]


if __name__ == "__main__":
    directory = "data/raw"
    max_chunk_size = 2000
    chunk_collector = ChunkCollector(directory, max_chunk_size)
    chunk_collector.collect_chunks()
    print(f"Collected {len(chunk_collector.get_chunks())} chunks.")
    all_texts = chunk_collector.fetch_all()
    print(f"Total text length: {sum(len(text) for text in all_texts)}")
