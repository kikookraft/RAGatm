
class BaseChunker:
    def __init__(self, chunk_size: int):
        self.content: str = ""
        self.chunk_size = chunk_size

    def chunk(self, data: str) -> list[str]:
        raise NotImplementedError("Subclasses should implement this method.")


class TextChunker(BaseChunker):
    def chunk(self, data: str) -> list[str]:
        self.content = data
        chunks: list[str] = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks


class PythonChunker(BaseChunker):
    def chunk(self, data: str) -> list[str]:
        self.content = data
        chunks: list[str] = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks


class ChunkerFactory:
    @staticmethod
    def get_chunker(file_extension: str, chunk_size: int) -> BaseChunker:
        if file_extension == ".py":
            return PythonChunker(chunk_size)
        else:
            return TextChunker(chunk_size)
